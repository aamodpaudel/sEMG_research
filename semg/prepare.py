"""Build the utterance index and preprocessed cache, and print dataset statistics.

    python -m semg.prepare
"""
import argparse
from collections import Counter

import numpy as np

from .train import load_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw/emg_data")
    ap.add_argument("--testset", default="data/testset_largedev.json")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args()
    utts = load_splits(args.data_root, args.testset, args.index, args.cache, args.rebuild_index)
    counts = Counter((u.split, "silent" if u.silent else "voiced", "closed" if u.closed_vocab else "open")
                     for u in utts)
    for k, v in sorted(counts.items()):
        print(f"{'/'.join(k):28s} {v}")
    lens = np.array([np.load(f"{args.cache}/{u.uid}.npy", mmap_mode="r").shape[0] for u in utts])
    print("hours %.1f | median %.1fs | max %.1fs | >16s: %d"
          % (lens.sum() / 3.6e6, np.median(lens) / 1000, lens.max() / 1000, (lens > 16000).sum()))
    evalset = [u for u in utts if u.split in ("dev", "test") and u.silent and not u.closed_vocab]
    print("unique words in silent dev+test:", len({w for u in evalset for w in u.text.split()}))
    print("closed-vocab examples:", [u.text for u in utts if u.closed_vocab][:3])


if __name__ == "__main__":
    main()
