"""Build the files Afono Nullpoint ships with, so it runs without the 3.9 GB dataset.

    python -m semg.bundle

Writes
  vocab/lm.json              unigram + bigram word counts from TRAINING transcripts only
  app/demo_pack/pack.npz     raw EMG of ~150 dataset recordings (offset removed, float16)
  app/demo_pack/pack.json    their transcripts and which split they come from
Test recordings are included for trying the app; calibration only ever uses dev recordings.
"""
import json
import os
from collections import Counter

import numpy as np

from .data import load_index
from .decode import bigram_counts

SETS = [("closed-test", True, "test", 43), ("closed-dev", True, "dev", 42),
        ("open-test", False, "test", 30), ("open-dev", False, "dev", 40)]


def main(index="data/index.json"):
    utts = load_index(index)
    train = [u for u in utts if u.split == "train"]
    lm = {}
    for name, closed in (("dates", True), ("open", False)):
        texts = [u.text for u in train if u.closed_vocab == closed]
        uni = Counter(w for t in texts for w in t.split())
        bi = bigram_counts(texts)
        lm[name] = {"unigram": dict(uni), "bigram": [[a, b, c] for (a, b), c in bi.items()]}
    os.makedirs("vocab", exist_ok=True)
    with open("vocab/lm.json", "w", encoding="utf-8") as f:
        json.dump(lm, f)

    os.makedirs("app/demo_pack", exist_ok=True)
    arrays, meta = {}, []
    for name, closed, split, n in SETS:
        pick = sorted((u for u in utts if u.split == split and u.silent and u.closed_vocab == closed), key=lambda u: u.path)[:n]
        for u in pick:
            x = np.load(u.path + "_emg.npy").astype(np.float64)
            med = np.median(x, axis=0)
            arrays[u.uid] = (x - med).astype(np.float16)
            arrays[u.uid + "_offset"] = med.astype(np.float32)
            meta.append({"uid": u.uid, "text": u.text, "set": name, "seconds": round(x.shape[0] / 1000, 2)})
    np.savez_compressed("app/demo_pack/pack.npz", **arrays)
    with open("app/demo_pack/pack.json", "w", encoding="utf-8") as f:
        json.dump({"source": "Gaddy & Klein 2020, CC BY 4.0, doi:10.5281/zenodo.4064408", "recordings": meta}, f)
    print(f"lm.json {os.path.getsize('vocab/lm.json') / 1e6:.1f} MB, demo pack {len(meta)} recordings, "
          f"{os.path.getsize('app/demo_pack/pack.npz') / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
