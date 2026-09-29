"""Benchmark a trained model on silent speech (protocol v2, see METRICS.md).

Tasks (all on SILENT test recordings the model never trained on):
  dates      closed vocabulary of dates & times (lexicon = its training words)
  closed-N   test sentences whose words fit an N-word vocabulary, decoded with exactly N words
  open       any sentence; lexicon = every training word (out-of-vocabulary words count as errors)
Decoder settings are tuned on the dev split only. Every number gets a 95% bootstrap CI.

    python -m semg.evaluate --ckpt runs/base/best.pt --out runs/base/report.json
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from collections import Counter

import numpy as np
import torch

from .data import EMGDataset, load_index
from .decode import Lexicon, greedy, lexicon_beam_search
from .metrics import score
from .model import load_model
from .registry import headline_metrics, log_run
from .text import wer
from .train import run_model

PROTOCOL = "v2"


def closed_subset(utts, n_words, train_counts):
    """Keep test sentences (in index order) while their combined vocabulary stays <= n_words,
    then pad the vocabulary with the most frequent training words up to exactly n_words."""
    vocab, keep = set(), []
    for u in utts:
        new = vocab | set(u.text.split())
        if len(new) <= n_words:
            vocab, keep = new, keep + [u]
    for w, _ in train_counts.most_common():
        if len(vocab) >= n_words:
            break
        vocab.add(w)
    return keep, vocab


def decode_timed(lps, lex, **kw):
    hyps, ms = [], []
    for lp in lps:
        t0 = time.perf_counter()
        hyps.append(lexicon_beam_search(lp, lex, **kw))
        ms.append((time.perf_counter() - t0) * 1000)
    return hyps, ms


def tune(lps, refs, lex, grid_lm=(0.0, 0.5, 1.0), grid_bonus=(0.0, 1.0, 2.0, 3.0), beam=16):
    best = None
    for lm_w, bonus in itertools.product(grid_lm, grid_bonus):
        w = wer(refs, [lexicon_beam_search(lp, lex, lm_weight=lm_w, word_bonus=bonus, beam=beam) for lp in lps])
        if best is None or w < best[0]:
            best = (w, lm_w, bonus)
    return {"lm_weight": best[1], "word_bonus": best[2], "dev_wer": best[0]}


def model_ms_per_utt(model, utts, cache, device, limit=20):
    """Wall-clock encoder time per utterance, batch size 1 (what a live user experiences)."""
    ms = []
    model.eval()
    with torch.no_grad():
        for u in utts[:limit]:
            x = torch.from_numpy(np.load(os.path.join(cache, u.uid + ".npy")).astype(np.float32))[None].to(device)
            model(x)  # warm-up
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ms)) if ms else None


def split_hash(utts):
    return hashlib.sha256("\n".join(sorted(u.uid for u in utts)).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--vocab-sizes", default="100,250,500")
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    utts = load_index(args.index)
    train = [u for u in utts if u.split == "train"]
    open_counts = Counter(w for u in train if not u.closed_vocab for w in u.text.split())
    date_counts = Counter(w for u in train if u.closed_vocab for w in u.text.split())
    sets = {name: [u for u in utts if u.split == split and u.silent and u.closed_vocab == closed]
            for name, split, closed in [("dev", "dev", False), ("test", "test", False),
                                        ("dates_dev", "dev", True), ("dates_test", "test", True)]}
    lps = {k: run_model(model, EMGDataset(v, args.cache), device) if v else [] for k, v in sets.items()}
    secs = {k: [lp.shape[0] / 125.0 for lp in v] for k, v in lps.items()}  # 125 output frames per second
    enc_ms = model_ms_per_utt(model, sets["test"], args.cache, device)

    open_lex = Lexicon(open_counts.keys(), open_counts)
    dates_lex = Lexicon(date_counts.keys(), date_counts)
    params = tune(lps["dev"][:100], [u.text for u in sets["dev"][:100]], open_lex)
    kw = dict(beam=args.beam, lm_weight=params["lm_weight"], word_bonus=params["word_bonus"])

    tasks, predictions = {}, []

    def run_task(name, task_utts, task_lps, lex, vocab_size):
        refs = [u.text for u in task_utts]
        g = [greedy(lp) for lp in task_lps]
        h, dec_ms = decode_timed(task_lps, lex, **kw)
        s = [lp.shape[0] / 125.0 for lp in task_lps]
        compute = [d + (enc_ms or 0) for d in dec_ms]
        tasks[name] = {"lexicon": score(refs, h, s, vocab_size, compute), "greedy": score(refs, g, s),
                       "decode_ms_median": float(np.median(dec_ms)) if dec_ms else None}
        for u, r, gg, hh, ss in zip(task_utts, refs, g, h, s):
            predictions.append({"task": name, "uid": u.uid, "seconds": round(ss, 3), "ref": r, "greedy": gg, "hyp": hh})

    run_task("dates", sets["dates_test"], lps["dates_test"], dates_lex, len(dates_lex))
    for n in [int(x) for x in args.vocab_sizes.split(",")]:
        keep, vocab = closed_subset(sets["test"], n, open_counts)
        idx = {u.uid: i for i, u in enumerate(sets["test"])}
        run_task(f"closed{n}", keep, [lps["test"][idx[u.uid]] for u in keep], Lexicon(vocab, open_counts), n)
    run_task("open", sets["test"], lps["test"], open_lex, len(open_lex))
    oov = sum(w not in open_lex.logp for u in sets["test"] for w in u.text.split()) / max(
        1, sum(len(u.text.split()) for u in sets["test"]))

    report = {
        "protocol": PROTOCOL, "ckpt": args.ckpt, "epoch": ckpt.get("epoch"), "params_m": round(
            sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "dataset": {"name": "Gaddy & Klein 2020 silent speech EMG", "doi": "10.5281/zenodo.4064408",
                    "speakers": 1, "channels": model.config["n_channels"], "sample_rate_hz": 1000,
                    "test_split": "testset_largedev.json (official) for books; 10% hash split for dates (ours)",
                    "split_hash": {k: split_hash(v) for k, v in sets.items()},
                    "n": {k: len(v) for k, v in sets.items()}, "open_test_oov_rate": oov},
        "decoder": {"type": "lexicon-constrained CTC prefix beam search + unigram prior", **kw,
                    "tuned_on": "dev (first 100 utterances)", "dev_wer_at_tuning": params["dev_wer"]},
        "compute": {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                    "encoder_ms_per_utterance_median": enc_ms, "torch": torch.__version__, "python": platform.python_version()},
        "tasks": tasks,
    }
    # backward-compatible keys used by the registry headline
    report["closed_vocab_dates"] = {"wer": tasks["dates"]["lexicon"]["wer"]}
    report["closed_n"] = {n: {"word_accuracy": tasks[f"closed{n}"]["lexicon"]["word_accuracy"],
                              "sentences": tasks[f"closed{n}"]["lexicon"]["n_utterances"]}
                          for n in [int(x) for x in args.vocab_sizes.split(",")]}
    report["open_lexicon"] = {"wer": tasks["open"]["lexicon"]["wer"]}
    report["open_greedy"] = {"cer": tasks["open"]["greedy"]["cer"], "n": tasks["open"]["greedy"]["n_utterances"]}

    print(json.dumps({k: v for k, v in report.items() if k != "tasks"}, indent=2))
    for name, t in tasks.items():
        L = t["lexicon"]
        print(f"{name:10s} n={L['n_utterances']:3d} words={L['n_words']:4d}  word acc {100*L['word_accuracy']:5.1f}% "
              f"[{100*L['word_accuracy_ci95'][0]:.1f}, {100*L['word_accuracy_ci95'][1]:.1f}]  "
              f"ITR {L.get('itr_bits_per_min', 0):6.1f} bits/min  greedy CER {100*t['greedy']['cer']:.1f}%")
    run_id = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    pred_path = os.path.join(os.path.dirname(out), "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    summary = {name: {k: t["lexicon"].get(k) for k in ("word_accuracy", "word_accuracy_ci95", "wer", "wer_ci95",
                                                        "n_utterances", "n_words", "itr_bits_per_min",
                                                        "correct_wpm", "real_time_factor", "vocab_size")}
               for name, t in tasks.items()}
    log_run("evaluate", run_id, ckpt=args.ckpt, report=out, predictions=pred_path, protocol=PROTOCOL,
            metrics=headline_metrics(report), tasks=summary, n_test=len(sets["test"]),
            closed_n_sentences={k: v["sentences"] for k, v in report["closed_n"].items()}, notes=args.notes)
    try:
        from .site_data import build
        build()
    except Exception as e:  # the site is optional
        print("site data not updated:", e)


if __name__ == "__main__":
    main()
