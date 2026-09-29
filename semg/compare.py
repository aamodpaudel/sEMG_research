"""Paired comparison of two checkpoints on the same held-out recordings.

    python -m semg.compare --profile demo-open --a runs/base/best.pt --b runs/calib-.../best.pt

Reports WER of each, the difference with a 95% paired-bootstrap CI, and a one-sided
p-value for "B is not better than A". Logged to the registry as kind 'compare'.
"""
import argparse
import os

import torch

from .data import EMGDataset, build_cache, load_index
from .decode import Lexicon, lexicon_beam_search
from .metrics import paired_bootstrap
from .model import load_model
from .registry import log_run
from .train import run_model
from .user import Profile


def holdout_set(profile_name, index="data/index.json", which="profile"):
    if which == "profile":
        p = Profile(profile_name)
        utts = [u for u in p.utterances() if u.split == "dev"]
        counts = p.word_counts()
    else:
        raise ValueError(which)
    return utts, Lexicon(counts.keys(), counts)


def compare(profile, a, b, cache="data/cache", log=True, notes=""):
    utts, lex = holdout_set(profile)
    build_cache(utts, cache, workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refs = [u.text for u in utts]
    hyps = {}
    for name, ck in (("a", a), ("b", b)):
        model, _ = load_model(ck, device)
        lps = run_model(model, EMGDataset(utts, cache), device)
        hyps[name] = [lexicon_beam_search(lp, lex, beam=16) for lp in lps]
    res = paired_bootstrap(refs, hyps["a"], hyps["b"])
    res.update({"a": a, "b": b, "profile": profile, "lexicon_size": len(lex)})
    if log:
        log_run("compare", os.path.basename(os.path.dirname(os.path.abspath(b))), **res, notes=notes)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--a", required=True, help="baseline checkpoint")
    ap.add_argument("--b", required=True, help="candidate checkpoint")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    r = compare(args.profile, args.a, args.b, notes=args.notes)
    print(f"WER A {100*r['wer_a']:.1f}%  B {100*r['wer_b']:.1f}%  delta {100*r['delta_wer']:+.1f} pts "
          f"[{100*r['delta_ci95'][0]:+.1f}, {100*r['delta_ci95'][1]:+.1f}]  p(B not better) = {r['p_b_not_better']:.3f}  "
          f"(n = {r['n_utterances']})")


if __name__ == "__main__":
    main()


def compare_predictions(a_path, b_path, tasks=None, log=True, notes=""):
    """Paired test between two predictions.jsonl files from semg.evaluate (same test utterances)."""
    import json
    def load(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    A, B = load(a_path), load(b_path)
    out = {}
    for task in sorted({r["task"] for r in A} if tasks is None else tasks):
        a = {r["uid"]: r for r in A if r["task"] == task}
        b = {r["uid"]: r for r in B if r["task"] == task}
        ids = [u for u in a if u in b]
        res = paired_bootstrap([a[u]["ref"] for u in ids], [a[u]["hyp"] for u in ids], [b[u]["hyp"] for u in ids])
        out[task] = res
        if log:
            log_run("compare", f"{os.path.basename(os.path.dirname(os.path.abspath(b_path)))}:{task}", a=a_path, b=b_path,
                    task=task, **res, notes=notes)
    return out
