"""Use the parallel voiced recordings: DTW-align silent EMG to voiced EMG (after Gaddy & Klein 2020).

For every silent TRAINING utterance whose sentence was also recorded voiced (training split only;
voiced copies of dev/test sentences are not in the index), a teacher checkpoint encodes both
recordings; dynamic time warping on the cosine distance between the encoder features aligns each
silent frame to voiced frames; the target for a silent frame is the teacher's character posterior
on the VOICED recording at the aligned frames (voiced EMG is much easier to recognise).
Training then adds KL(target || model) on silent frames (`semg.train --align-weight`).

    python -m semg.align --teacher runs/base/best.pt --out data/align
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from .data import EMGDataset, load_index
from .model import load_model


def dtw_path(cost):
    """Classic DTW (steps (1,1), (1,0), (0,1)) on a (N, M) cost matrix; returns list of (i, j)."""
    N, M = cost.shape
    D = np.empty((N, M))
    C = np.cumsum(cost[0])
    D[0] = C
    for i in range(1, N):
        prev = D[i - 1]
        a = cost[i] + np.minimum(prev, np.concatenate([[np.inf], prev[:-1]]))   # diagonal or vertical
        c = np.cumsum(cost[i])
        # horizontal moves: D[i,j] = min_k<=j (a[k] + c[j] - c[k])
        D[i] = c + np.minimum.accumulate(a - c)
    i, j, path = N - 1, M - 1, [(N - 1, M - 1)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])))
            i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
        path.append((i, j))
    return path[::-1]


@torch.no_grad()
def encode_all(model, utts, cache, device):
    ds = EMGDataset(utts, cache)
    feats, posts = [], []
    model.eval()
    for i in range(len(ds)):
        x = torch.from_numpy(ds.load(i))[None].to(device)
        mode = torch.tensor([0 if utts[i].silent else 1], device=device)
        h, _ = model.encode(x, None, mode)
        lp = model.head(h).log_softmax(-1)
        f = torch.nn.functional.normalize(h[0].float(), dim=-1)
        feats.append(f.cpu().numpy())
        posts.append(lp[0].float().exp().cpu().numpy())
    return feats, posts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/base/best.pt")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="data/align")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.teacher, device)
    utts = [u for u in load_index(args.index) if u.split == "train" and not u.closed_vocab]
    voiced = defaultdict(list)
    for u in utts:
        if not u.silent:
            voiced[(u.book, u.sentence_index)].append(u)
    pairs = [(u, voiced[(u.book, u.sentence_index)][0]) for u in utts
             if u.silent and voiced.get((u.book, u.sentence_index))]
    print(f"{len(pairs)} silent training utterances with a voiced parallel recording")
    fs, ps = encode_all(model, [p[0] for p in pairs], args.cache, device)
    fv, pv = encode_all(model, [p[1] for p in pairs], args.cache, device)
    agree = []
    for (s, v), a, b, qs, qv in zip(pairs, fs, fv, ps, pv):
        path = dtw_path(1.0 - a @ b.T)
        tgt = np.zeros_like(qs)
        cnt = np.zeros(len(qs))
        for i, j in path:
            tgt[i] += qv[j]
            cnt[i] += 1
        tgt /= cnt[:, None]
        np.save(os.path.join(args.out, s.uid + ".npy"), tgt.astype(np.float16))
        agree.append(float((tgt.argmax(1) == qs.argmax(1)).mean()))
    print(f"saved {len(pairs)} targets to {args.out}; frame argmax agreement silent vs aligned voiced: {np.mean(agree):.3f}")


if __name__ == "__main__":
    main()
