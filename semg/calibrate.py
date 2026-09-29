"""Supervised per-user calibration: fine-tune a trained model on a profile's labelled recordings.

    python -m semg.calibrate --profile default --base runs/base/best.pt

- Mixes in "replay" utterances from the original training set so the model does not forget.
- Measures the profile's held-out recordings before and after, and logs both to the registry.
- `--import-dataset closed-dev` fills a demo profile with real dataset recordings so the loop
  can be tested before any hardware exists.
"""
import argparse
import datetime as _dt
import os
import random
import time

import numpy as np
import torch

from .data import EMGDataset, build_cache, load_index
from .decode import Lexicon, greedy, lexicon_beam_search
from .model import adapt_channels, load_model
from .registry import log_run
from .text import cer, wer
from .train import run_model, train
from .user import Profile


def measure(model, utts, cache, device, lexicon):
    if not utts:
        return None
    lps = run_model(model, EMGDataset(utts, cache), device)
    refs = [u.text for u in utts]
    g = [greedy(lp) for lp in lps]
    lx = [lexicon_beam_search(lp, lexicon, beam=16) for lp in lps]
    return {"n": len(utts), "greedy_wer": wer(refs, g), "greedy_cer": cer(refs, g),
            "lexicon_wer": wer(refs, lx), "lexicon_word_acc": 1 - wer(refs, lx)}


def import_dataset(profile: Profile, index_path: str, which: str, limit: int):
    """Copy real dataset recordings + transcripts into a profile (demo without hardware)."""
    utts = load_index(index_path)
    if which == "closed-dev":
        pick = [u for u in utts if u.closed_vocab and u.silent and u.split == "dev"]
    elif which == "open-dev":
        pick = [u for u in utts if not u.closed_vocab and u.silent and u.split == "dev"]
    else:
        raise ValueError(which)
    for u in pick[:limit]:
        rid = "ds" + u.uid[:10]
        if not profile.has(rid):
            profile.add_recording(np.load(u.path + "_emg.npy"), rid)
            profile.add_label(rid, u.text, source="prompted")
    return len(pick[:limit])


def calibrate(profile_name="default", base="runs/base/best.pt", out=None, epochs=15, lr=1e-4,
              replay=1.0, index="data/index.json", cache="data/cache", notes="", log=print):
    profile = Profile(profile_name)
    utts = profile.utterances()
    tr = [u for u in utts if u.split == "train"]
    ho = [u for u in utts if u.split == "dev"]
    if len(tr) < 5 or len(ho) < 1:
        raise RuntimeError(f"need at least 5 training and 1 held-out labelled recordings, have {len(tr)}/{len(ho)}")
    build_cache(utts, cache, workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(base, device)
    channels = {int(np.load(u.path + "_emg.npy", mmap_mode="r").shape[1]) for u in utts}
    if len(channels) != 1:
        raise RuntimeError(f"recordings in this profile have different channel counts: {sorted(channels)}")
    n_ch = channels.pop()
    if n_ch != model.config["n_channels"]:
        log(f"adapting model input from {model.config['n_channels']} to {n_ch} channels")
        model = adapt_channels(model, n_ch)

    base_train = [u for u in load_index(index) if u.split == "train"]
    counts = profile.word_counts()           # personal vocabulary = words this user has confirmed
    lexicon = Lexicon(counts.keys(), counts)
    before = measure(model, ho, cache, device, lexicon)
    log({"before": before})

    rng = random.Random(0)
    replay_utts = rng.sample(base_train, min(len(base_train), int(len(tr) * replay))) if replay > 0 else []
    run_id = "calib-%s-%s" % (profile_name, _dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out = out or os.path.join("runs", run_id)
    t0 = time.time()
    model.train()
    train(model, EMGDataset(tr + replay_utts, cache, train=True), EMGDataset(ho, cache), out,
          epochs=epochs, lr=lr, max_samples=150_000, device=device.type, eval_every=max(1, epochs // 3),
          workers=0, log=log)
    model, _ = load_model(os.path.join(out, "best.pt"), device)
    after = measure(model, ho, cache, device, lexicon)
    log({"after": after})
    from .compare import compare
    paired = compare(profile_name, base, os.path.join(out, "best.pt"), cache=cache, log=False)
    log({"paired": paired})
    rec = log_run("calibrate", os.path.basename(out), out=out, base=base, profile=profile.summary(),
                  train_recordings=len(tr), holdout_recordings=len(ho), replay_utterances=len(replay_utts),
                  epochs=epochs, lr=lr, before=before, after=after, paired=paired, minutes=round((time.time() - t0) / 60, 1),
                  notes=notes)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default")
    ap.add_argument("--base", default="runs/base/best.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replay", type=float, default=1.0, help="original-data utterances per user utterance")
    ap.add_argument("--import-dataset", choices=["closed-dev", "open-dev"], default=None)
    ap.add_argument("--import-limit", type=int, default=1000)
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    if args.import_dataset:
        n = import_dataset(Profile(args.profile), args.index, args.import_dataset, args.import_limit)
        print(f"imported {n} dataset recordings into profile '{args.profile}'")
    rec = calibrate(args.profile, args.base, args.out, args.epochs, args.lr, args.replay, args.index,
                    notes=args.notes)
    print("before:", rec["before"])
    print("after: ", rec["after"])
    try:
        from .site_data import build
        build()
    except Exception as e:
        print("site data not updated:", e)


if __name__ == "__main__":
    main()
