"""Benchmark a trained model on silent speech (protocol v3, see METRICS.md).

Tasks (all on SILENT test recordings the model never trained on):
  dates        closed vocabulary of dates & times (lexicon = its training words)
  closed-N     test sentences whose words fit an N-word vocabulary, decoded with exactly N words
  open         any sentence; lexicon = every training word (out-of-vocabulary words count as errors)
  pageN-easy   per utterance: its distinct words + hashed training-word distractors (~N words),
  pageN-hard   ... or + nearest-by-edit-distance distractors; uniform prior over the page words
  page30-hard+lm  secondary: page30-hard with the word n-gram prior renormalised over the page words
  isoN-easy/hard  isolated-word proxy (oracle word spans from forced alignment, N candidates)
Decoder settings are tuned on the dev split only (same constructions on dev utterances).
Every number gets a 95% bootstrap CI.

    python -m semg.evaluate --ckpt runs/base/best.pt --lm trigram
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
from .decode import Lexicon, bigram_counts, greedy, lexicon_beam_search
from .isoword import isolated_word_accuracy
from .lm import NGramLM
from .metrics import score
from .model import load_model
from .page import NearestIndex, page_lexicon
from .registry import headline_metrics, log_run
from .text import CHAR_TO_ID, SPACE, wer
from .train import run_model

PROTOCOL = "v3"
PAGE_SIZES = (10, 20, 30)
GRID_LM = (0.0, 0.5, 1.0, 1.5, 2.0)
GRID_BONUS = (0.0, 1.0, 2.0, 3.0, 4.0)
GRID_PAGE_BONUS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
N_TUNE = 100


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


def decode_timed(lps, lexs, **kw):
    """lexs: one Lexicon for all, or a list with one per utterance."""
    hyps, ms = [], []
    for i, lp in enumerate(lps):
        lex = lexs[i] if isinstance(lexs, list) else lexs
        t0 = time.perf_counter()
        hyps.append(lexicon_beam_search(lp, lex, **kw))
        ms.append((time.perf_counter() - t0) * 1000)
    return hyps, ms


def tune(lps, refs, lexs, grid_lm=GRID_LM, grid_bonus=GRID_BONUS, beam=16):
    best = None
    for lm_w, bonus in itertools.product(grid_lm, grid_bonus):
        hyps, _ = decode_timed(lps, lexs, lm_weight=lm_w, word_bonus=bonus, beam=beam)
        w = wer(refs, hyps)
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


def units_spec(units):
    if units == "char":
        return (lambda w: [CHAR_TO_ID[c] for c in w]), SPACE
    from .phonemes import SEP, TOKEN_TO_ID, pronounce
    return (lambda w: [TOKEN_TO_ID[p] for p in pronounce(w)]), SEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--vocab-sizes", default="100,250,500")
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lm", choices=["unigram", "bigram", "trigram"], default="trigram",
                    help="word prior for dates/closed/open (training transcripts only)")
    ap.add_argument("--units", choices=["char", "phone"], default="char",
                    help="decode through characters, or through the pronunciation lexicon (phoneme head)")
    ap.add_argument("--tag", default=None, help="suffix for report_<tag>.json / predictions_<tag>.jsonl")
    ap.add_argument("--no-iso", action="store_true", help="skip the isolated-word proxy")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    if args.units == "phone" and not model.config.get("n_phones"):
        raise SystemExit("this checkpoint has no phoneme head")
    spell, space = units_spec(args.units)
    utts = load_index(args.index)
    train = [u for u in utts if u.split == "train"]
    open_texts = [u.text for u in train if not u.closed_vocab]
    date_texts = [u.text for u in train if u.closed_vocab]
    open_counts = Counter(w for t in open_texts for w in t.split())
    date_counts = Counter(w for t in date_texts for w in t.split())
    sets = {name: [u for u in utts if u.split == split and u.silent and u.closed_vocab == closed]
            for name, split, closed in [("dev", "dev", False), ("test", "test", False),
                                        ("dates_dev", "dev", True), ("dates_test", "test", True)]}
    head = "char" if args.units == "char" else "both"
    raw = {k: run_model(model, EMGDataset(v, args.cache), device, head=head) if v else [] for k, v in sets.items()}
    if head == "both":
        char_lps = {k: [a for a, _ in v] for k, v in raw.items()}
        lps = {k: [b for _, b in v] for k, v in raw.items()}
    else:
        char_lps = lps = raw
    enc_ms = model_ms_per_utt(model, sets["test"], args.cache, device)

    def make_lex(words, counts, texts):
        kw = dict(spell=spell, space=space) if args.units == "phone" else {}
        if args.lm == "bigram":
            return Lexicon(words, counts, bigram_counts(texts), **kw)
        if args.lm == "trigram":
            return Lexicon(words, counts, lm=LMS[id(texts)], **kw)
        return Lexicon(words, counts, **kw)

    LMS = {}
    if args.lm == "trigram":
        LMS[id(open_texts)] = NGramLM(open_texts, 3)
        LMS[id(date_texts)] = NGramLM(date_texts, 3)
    open_lex = make_lex(open_counts.keys(), open_counts, open_texts)
    dates_lex = make_lex(date_counts.keys(), date_counts, date_texts)
    t0 = time.time()
    params = tune(lps["dev"][:N_TUNE], [u.text for u in sets["dev"][:N_TUNE]], open_lex)
    print("tuned open decoder on dev:", params, "%.0f s" % (time.time() - t0))
    kw = dict(beam=args.beam, lm_weight=params["lm_weight"], word_bonus=params["word_bonus"])

    tasks, predictions = {}, []

    def run_task(name, task_utts, task_lps, task_char_lps, lex, vocab_size, dkw):
        refs = [u.text for u in task_utts]
        g = [greedy(lp) for lp in task_char_lps]
        h, dec_ms = decode_timed(task_lps, lex, **dkw)
        s = [lp.shape[0] / 125.0 for lp in task_lps]
        compute = [d + (enc_ms or 0) for d in dec_ms]
        tasks[name] = {"lexicon": score(refs, h, s, vocab_size, compute), "greedy": score(refs, g, s),
                       "decode_ms_median": float(np.median(dec_ms)) if dec_ms else None,
                       "decoder": {k: v for k, v in dkw.items()}}
        for u, r, gg, hh, ss in zip(task_utts, refs, g, h, s):
            predictions.append({"task": name, "uid": u.uid, "seconds": round(ss, 3), "ref": r, "greedy": gg, "hyp": hh})
        L = tasks[name]["lexicon"]
        print(f"{name:14s} n={L['n_utterances']:3d} words={L['n_words']:4d}  word acc {100*L['word_accuracy']:5.1f}% "
              f"[{100*L['word_accuracy_ci95'][0]:.1f}, {100*L['word_accuracy_ci95'][1]:.1f}]", flush=True)

    run_task("dates", sets["dates_test"], lps["dates_test"], char_lps["dates_test"], dates_lex, len(dates_lex), kw)
    idx = {u.uid: i for i, u in enumerate(sets["test"])}
    for n in [int(x) for x in args.vocab_sizes.split(",")]:
        keep, vocab = closed_subset(sets["test"], n, open_counts)
        run_task(f"closed{n}", keep, [lps["test"][idx[u.uid]] for u in keep],
                 [char_lps["test"][idx[u.uid]] for u in keep], make_lex(vocab, open_counts, open_texts), n, kw)
    run_task("open", sets["test"], lps["test"], char_lps["test"], open_lex, len(open_lex), kw)
    oov = sum(w not in open_lex.vocab for u in sets["test"] for w in u.text.split()) / max(
        1, sum(len(u.text.split()) for u in sets["test"]))

    # ---- page-vocabulary tasks (protocol v3) --------------------------------------------------
    pool = sorted(open_counts)
    near = NearestIndex(pool)
    page_info = {}
    lex_kw = dict(spell=spell, space=space) if args.units == "phone" else {}
    for n in PAGE_SIZES:
        for variant in ("easy", "hard"):
            name = f"page{n}-{variant}"
            dev_u, test_u = sets["dev"][:N_TUNE], sets["test"]
            dev_lex = [Lexicon(page_lexicon(u.uid, u.text, n, pool, variant, near), **lex_kw) for u in dev_u]
            test_words = [page_lexicon(u.uid, u.text, n, pool, variant, near) for u in test_u]
            test_lex = [Lexicon(w, **lex_kw) for w in test_words]
            p = tune(lps["dev"][:N_TUNE], [u.text for u in dev_u], dev_lex, grid_lm=(0.0,), grid_bonus=GRID_PAGE_BONUS)
            run_task(name, test_u, lps["test"], char_lps["test"], test_lex, n,
                     dict(beam=args.beam, lm_weight=0.0, word_bonus=p["word_bonus"]))
            sizes = [len(w) for w in test_words]
            page_info[name] = {"tuned": p, "lexicon_size_mean": float(np.mean(sizes)),
                               "lexicon_size_max": int(max(sizes)), "lexicon_size_min": int(min(sizes))}
            tasks[name]["lexicon"]["vocab_size"] = int(round(np.mean(sizes)))
    # secondary: page30-hard with the word prior (renormalised over the page words)
    if args.lm != "unigram":
        name = "page30-hard+lm"
        dev_u = sets["dev"][:N_TUNE]
        dev_lex = [make_lex(page_lexicon(u.uid, u.text, 30, pool, "hard", near), open_counts, open_texts) for u in dev_u]
        test_lex = [make_lex(page_lexicon(u.uid, u.text, 30, pool, "hard", near), open_counts, open_texts)
                    for u in sets["test"]]
        p = tune(lps["dev"][:N_TUNE], [u.text for u in dev_u], dev_lex, grid_lm=(0.5, 1.0, 1.5),
                 grid_bonus=(-1.0, 0.0, 1.0, 2.0, 3.0))
        run_task(name, sets["test"], lps["test"], char_lps["test"], test_lex, 30,
                 dict(beam=args.beam, lm_weight=p["lm_weight"], word_bonus=p["word_bonus"]))
        page_info[name] = {"tuned": p}

    # ---- isolated-word proxy -------------------------------------------------------------------
    iso = {}
    if not args.no_iso:
        dpool = sorted(date_counts)
        dnear = NearestIndex(dpool)
        for n in PAGE_SIZES:
            for variant in ("easy", "hard"):
                iso[f"iso{n}-{variant}"] = isolated_word_accuracy(sets["test"], lps["test"], spell, space, n, variant, pool, near)
                if n <= len(dpool):
                    iso[f"iso{n}-{variant}-dates"] = isolated_word_accuracy(sets["dates_test"], lps["dates_test"], spell,
                                                                            space, n, variant, dpool, dnear)
        for k, v in iso.items():
            print(f"{k:18s} acc {100*v['accuracy']:5.1f}% [{100*v['accuracy_ci95'][0]:.1f}, {100*v['accuracy_ci95'][1]:.1f}] "
                  f"words={v['n_words']} align_fail={v['alignment_failures']}")

    report = {
        "protocol": PROTOCOL, "ckpt": args.ckpt, "epoch": ckpt.get("epoch"), "params_m": round(
            sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "dataset": {"name": "Gaddy & Klein 2020 silent speech EMG", "doi": "10.5281/zenodo.4064408",
                    "speakers": 1, "channels": model.config["n_channels"], "sample_rate_hz": 1000,
                    "test_split": "testset_largedev.json (official) for books; 10% hash split for dates (ours)",
                    "split_hash": {k: split_hash(v) for k, v in sets.items()},
                    "n": {k: len(v) for k, v in sets.items()}, "open_test_oov_rate": oov},
        "decoder": {"type": f"lexicon-constrained CTC prefix beam search ({args.units}) + {args.lm} word prior",
                    "lm": args.lm, "units": args.units, **kw,
                    "tuned_on": f"dev (first {N_TUNE} utterances)", "dev_wer_at_tuning": params["dev_wer"]},
        "compute": {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                    "encoder_ms_per_utterance_median": enc_ms, "torch": torch.__version__, "python": platform.python_version()},
        "tasks": tasks, "page": page_info, "isolated_word_proxy": iso,
    }
    # backward-compatible keys used by the registry headline
    report["closed_vocab_dates"] = {"wer": tasks["dates"]["lexicon"]["wer"]}
    report["closed_n"] = {n: {"word_accuracy": tasks[f"closed{n}"]["lexicon"]["word_accuracy"],
                              "sentences": tasks[f"closed{n}"]["lexicon"]["n_utterances"]}
                          for n in [int(x) for x in args.vocab_sizes.split(",")]}
    report["open_lexicon"] = {"wer": tasks["open"]["lexicon"]["wer"]}
    report["open_greedy"] = {"cer": tasks["open"]["greedy"]["cer"], "n": tasks["open"]["greedy"]["n_utterances"]}

    run_id = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    tag = args.tag or (args.lm + ("_phone" if args.units == "phone" else ""))
    base = os.path.dirname(args.out) if args.out else os.path.dirname(args.ckpt)
    out = args.out or os.path.join(base, f"report_{tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    pred_path = os.path.join(base, f"predictions_{tag}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    summary = {name: {k: t["lexicon"].get(k) for k in ("word_accuracy", "word_accuracy_ci95", "wer", "wer_ci95",
                                                        "n_utterances", "n_words", "itr_bits_per_min",
                                                        "correct_wpm", "real_time_factor", "vocab_size")}
               for name, t in tasks.items()}
    log_run("evaluate", run_id, ckpt=args.ckpt, report=out, predictions=pred_path, protocol=PROTOCOL, lm=args.lm,
            units=args.units, metrics=headline_metrics(report), tasks=summary,
            isolated_word_proxy={k: {"accuracy": v["accuracy"], "accuracy_ci95": v["accuracy_ci95"]} for k, v in iso.items()},
            n_test=len(sets["test"]),
            closed_n_sentences={k: v["sentences"] for k, v in report["closed_n"].items()}, notes=args.notes)
    print("report:", out)


if __name__ == "__main__":
    main()
