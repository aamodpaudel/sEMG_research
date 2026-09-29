"""End-to-end smoke test on synthetic EMG written in the Gaddy dataset's file format.

Each character is rendered as an 80 ms burst of noise with a character-specific
channel pattern, so a model that trains correctly can learn it quickly.
"""
import json
import os

import numpy as np
import pytest
import torch

from semg.data import EMGDataset, build_cache, index_dataset
from semg.decode import Lexicon, greedy, lexicon_beam_search
from semg.model import EMGCTC
from semg.text import CHARS, CHAR_TO_ID, normalize_text, wer
from semg.train import run_model, train

WORDS = "yes no stop go back open close up down left right help water call".split()


def _render(text, rng, patterns):
    chunks = [rng.normal(0, 0.05, (150, 8))]
    for c in text:
        n = 60 if c == " " else 80
        gain = patterns[CHAR_TO_ID[c]]
        chunks.append(rng.normal(0, 1, (n, 8)) * gain * np.hanning(n)[:, None] * 5)
    chunks.append(rng.normal(0, 0.05, (150, 8)))
    return (np.concatenate(chunks) * 100 + 500).astype(np.float32)


def _make_dataset(root, n=300, seed=0):
    rng = np.random.default_rng(seed)
    patterns = rng.uniform(0.1, 1.0, (len(CHARS), 8))
    split = {"dev": [], "test": []}
    for mode in ["silent_parallel_data", "voiced_parallel_data"]:
        d = os.path.join(root, mode, "session1")
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            text = " ".join(rng.choice(WORDS, size=rng.integers(1, 4)))
            np.save(os.path.join(d, f"{i}_emg.npy"), _render(text, rng, patterns))
            with open(os.path.join(d, f"{i}_info.json"), "w") as f:
                json.dump({"text": text.capitalize() + ".", "book": "synthetic", "sentence_index": i}, f)
    split["dev"] = [["synthetic", i] for i in range(0, 20)]
    split["test"] = [["synthetic", i] for i in range(20, 40)]
    with open(os.path.join(root, "split.json"), "w") as f:
        json.dump(split, f)


def test_text_normalisation():
    assert normalize_text("It's 4:30 on the 3rd!") == "it's four thirty on the third"


def test_lexicon_decoder_collapses_repeats():
    lp = np.full((20, len(CHARS)), -10.0)
    for t, c in enumerate("hhelp  waterr"):
        lp[t, CHAR_TO_ID[c]] = -0.05
    lp[13:, 0] = -0.01
    assert lexicon_beam_search(lp, Lexicon(WORDS)) == "help water"


def test_end_to_end(tmp_path):
    root = str(tmp_path / "emg_data")
    _make_dataset(root)
    utts = index_dataset(root, os.path.join(root, "split.json"))
    splits = {s: [u for u in utts if u.split == s] for s in ("train", "dev", "test")}
    assert len(splits["dev"]) == 20 and len(splits["test"]) == 20
    assert all(u.silent for u in splits["dev"] + splits["test"])  # voiced copies of test sentences dropped
    assert len(splits["train"]) == 2 * 300 - 80
    cache = str(tmp_path / "cache")
    build_cache(utts, cache, workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = EMGCTC(d_model=96, n_layers=2, n_heads=4, ff=192, dropout=0.1)
    hist = train(model, EMGDataset(splits["train"], cache, train=True), EMGDataset(splits["dev"], cache),
                 str(tmp_path / "run"), epochs=60, lr=2e-3, max_samples=60_000, device=device,
                 eval_every=20, workers=0, log=lambda r: None)
    assert hist[-1]["loss"] < hist[0]["loss"] * 0.5
    lps = run_model(model, EMGDataset(splits["test"], cache), torch.device(device))
    refs = [u.text for u in splits["test"]]
    g = wer(refs, [greedy(x) for x in lps])
    lx = wer(refs, [lexicon_beam_search(x, Lexicon(WORDS)) for x in lps])
    print(f"synthetic test WER greedy={g:.3f} lexicon={lx:.3f}")
    assert lx <= g + 1e-9 and lx < 0.3

    torch.save({"model": model.state_dict(), "config": model.config, "epoch": 30}, str(tmp_path / "m.pt"))
    from semg.export import export
    rep = export(str(tmp_path / "m.pt"), str(tmp_path / "m.onnx"), seconds=2.0)
    print(rep)
    assert rep["m.onnx"]["max_abs_logprob_diff"] < 1e-3


def test_profile_labels_and_channel_adapt(tmp_path):
    from semg.model import adapt_channels
    from semg.user import Profile
    p = Profile("t", root=str(tmp_path))
    rid = p.add_recording(np.random.randn(1200, 4).astype(np.float32))
    p.add_label(rid, "Open the menu", source="correction", model_output="open the many")
    p.add_label(rid, "open the menu please", source="correction")
    assert p.labels()[rid]["text"] == "open the menu please"          # latest label wins
    assert p.word_counts()["menu"] == 1 and p.summary()["labelled"] == 1
    with pytest.raises(ValueError):
        Profile("../escape", root=str(tmp_path))
    m4 = adapt_channels(EMGCTC(d_model=64, n_layers=1, n_heads=2, ff=128), 4)
    assert m4(torch.randn(1, 800, 4))[0].shape[-1] == len(CHARS)


def test_js_decoder_matches_python(tmp_path):
    import shutil
    import subprocess
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    lp = np.full((20, len(CHARS)), -10.0, dtype=np.float32)
    for t, c in enumerate("hhelp  waterr"):
        lp[t, CHAR_TO_ID[c]] = -0.05
    lp[13:, 0] = -0.01
    lp.tofile(tmp_path / "lp.bin")
    js = os.path.join(os.path.dirname(__file__), "..", "site", "demo", "afono-decode.js").replace("\\", "/")
    code = (f"const D=require('{js}');const fs=require('fs');const b=fs.readFileSync(process.argv[1]);"
            f"const lp=new Float32Array(b.buffer,b.byteOffset,b.length/4);"
            f"const chars={json.dumps(CHARS)};const lex=D.buildLexicon({json.dumps({w: 1 for w in WORDS})},chars);"
            f"console.log(D.lexiconBeam(lp,20,{len(CHARS)},lex,{{}}))")
    out = subprocess.run(["node", "-e", code, str(tmp_path / "lp.bin")], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == lexicon_beam_search(lp, Lexicon(WORDS)) == "help water"


def test_research_metrics():
    from semg.metrics import paired_bootstrap, score, wolpaw_bits
    assert abs(wolpaw_bits(2, 1.0) - 1.0) < 1e-9 and wolpaw_bits(51, 1 / 51) == 0.0
    s = score(["go back", "open menu"], ["go back", "open many"], [1.0, 1.0], 100, [5, 5])
    assert abs(s["wer"] - 0.25) < 1e-9 and s["wer_ci95"][0] <= s["wer"] <= s["wer_ci95"][1]
    assert abs(s["correct_wpm"] - 90.0) < 1e-6 and s["sentence_accuracy"] == 0.5
    p = paired_bootstrap(["a b", "c d"], ["x y", "c d"], ["a b", "c d"])
    assert p["delta_wer"] == -0.5 and 0.15 < p["p_b_not_better"] < 0.35  # ties (1 in 4 resamples) count as not better
