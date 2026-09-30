"""Train the EMG -> text CTC model.

    python -m semg.train --out runs/base
"""
import argparse
import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (EMGDataset, BucketSampler, build_cache, collate, index_dataset,
                   load_index, save_index)
from .decode import greedy
from .model import EMGCTC
from .registry import log_run
from .text import wer, cer


def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


@torch.no_grad()
def run_model(model, ds, device, max_samples=400_000, head="char"):
    """Returns list of (T', V) numpy log-prob arrays, in dataset order.
    head: "char", "phone" (needs a phoneme head) or "both" (list of (char, phone) pairs)."""
    model.eval()
    lengths = ds.compute_lengths()
    sampler = BucketSampler(lengths, max_samples, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)
    out = [None] * len(ds)
    order = [i for b in sampler for i in b]
    k = 0
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if head == "char":
                lp, ol = model(batch["emg"].to(device), batch["lengths"].to(device), batch["mode"].to(device))
                ph = None
            else:
                lp, ph, ol = model.forward_all(batch["emg"].to(device), batch["lengths"].to(device),
                                               batch["mode"].to(device))
        lp = lp.float().cpu().numpy()
        ph = ph.float().cpu().numpy() if ph is not None else None
        for j in range(lp.shape[0]):
            n = int(ol[j])
            out[order[k]] = (lp[j, :n] if head == "char" else ph[j, :n] if head == "phone"
                             else (lp[j, :n], ph[j, :n]))
            k += 1
    return out


def evaluate_greedy(model, ds, device):
    lps = run_model(model, ds, device)
    refs = [u.text for u in ds.utts]
    hyps = [greedy(lp) for lp in lps]
    return {"wer": wer(refs, hyps), "cer": cer(refs, hyps), "examples": list(zip(refs[:3], hyps[:3]))}


def train(model, train_ds, dev_ds, out_dir, epochs=80, lr=5e-4, max_samples=200_000,
          device="cuda", eval_every=5, workers=4, log=print, phone_weight=0.0, patience=0,
          weight_decay=0.01, align_weight=0.0):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)
    model.to(device)
    sampler = BucketSampler(train_ds.compute_lengths(), max_samples, shuffle=True)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                        worker_init_fn=_worker_init if workers else None,
                        persistent_workers=workers > 0)
    steps_per_epoch = len(sampler)
    total = epochs * steps_per_epoch
    warmup = max(1, min(1000, total // 10))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    best, step, history, since_best = float("inf"), 0, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lp, ph, ol = model.forward_all(batch["emg"].to(device), batch["lengths"].to(device),
                                               batch["mode"].to(device))
            loss = F.ctc_loss(lp.float().transpose(0, 1), batch["labels"].to(device), ol,
                              batch["label_lengths"].to(device), blank=0, zero_infinity=True)
            if phone_weight and ph is not None and "phones" in batch:
                loss = loss + phone_weight * F.ctc_loss(
                    ph.float().transpose(0, 1), batch["phones"].to(device), ol,
                    batch["phone_lengths"].to(device), blank=0, zero_infinity=True)
            if align_weight and "align" in batch and int(batch["align_lengths"].sum()) > 0:
                tg = batch["align"].to(device)
                n = torch.minimum(batch["align_lengths"].to(device), ol)
                T = min(tg.shape[1], lp.shape[1])
                m = (torch.arange(T, device=device)[None] < n[:, None]).float()
                kl = (tg[:, :T] * (torch.log(tg[:, :T] + 1e-6) - lp[:, :T].float())).sum(-1)
                loss = loss + align_weight * (kl * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
        rec = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "sec": round(time.time() - t0, 1)}
        if dev_ds is not None and (epoch % eval_every == 0 or epoch == epochs):
            ev = evaluate_greedy(model, dev_ds, device)
            rec.update(dev_wer=ev["wer"], dev_cer=ev["cer"])
            if ev["wer"] < best:
                best, since_best = ev["wer"], 0
                torch.save({"model": model.state_dict(), "config": model.config, "epoch": epoch,
                            "dev": ev}, os.path.join(out_dir, "best.pt"))
            else:
                since_best += 1
            rec["example"] = ev["examples"][0]
        torch.save({"model": model.state_dict(), "config": model.config, "epoch": epoch},
                   os.path.join(out_dir, "last.pt"))
        history.append(rec)
        with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(rec)
        if patience and since_best >= patience:
            log({"early_stop": epoch, "best_dev_wer": best})
            break
    return history


def load_splits(data_root, testset, index_path, cache_dir, rebuild=False):
    if rebuild or not os.path.exists(index_path):
        utts = index_dataset(data_root, testset)
        save_index(utts, index_path)
    utts = load_index(index_path)
    build_cache(utts, cache_dir)
    return utts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw/emg_data")
    ap.add_argument("--testset", default="data/testset_largedev.json")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-samples", type=int, default=200_000, help="padded samples per batch")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--phone-weight", type=float, default=0.0, help="weight of the auxiliary phoneme CTC loss (0 = off)")
    ap.add_argument("--aug", choices=["v1", "v2"], default="v1")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--align-dir", default=None, help="DTW-aligned voiced targets from semg.align")
    ap.add_argument("--align-weight", type=float, default=0.0)
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many evaluations without dev gain")
    ap.add_argument("--notes", default="", help="what this run is testing (goes in the registry)")
    args = ap.parse_args()

    utts = load_splits(args.data_root, args.testset, args.index, args.cache, args.rebuild_index)
    counts = Counter((u.split, "silent" if u.silent else "voiced", "closed" if u.closed_vocab else "open")
                     for u in utts)
    print("utterances:", dict(sorted(counts.items())))
    train_utts = [u for u in utts if u.split == "train"]
    dev_utts = [u for u in utts if u.split == "dev" and not u.closed_vocab]
    from .phonemes import N_TOKENS
    model = EMGCTC(d_model=args.d_model, n_layers=args.layers, ff=4 * args.d_model, dropout=args.dropout,
                   n_phones=N_TOKENS if args.phone_weight > 0 else 0)
    print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    t0 = time.time()
    ph = args.phone_weight > 0
    if args.init:
        own = model.state_dict()
        src = torch.load(args.init, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict({k: v for k, v in src.items() if k in own and own[k].shape == v.shape}, strict=False)
    hist = train(model, EMGDataset(train_utts, args.cache, train=True, phones=ph, aug=args.aug,
                                   align_dir=args.align_dir if args.align_weight > 0 else None),
                 EMGDataset(dev_utts, args.cache),
                 args.out, epochs=args.epochs, lr=args.lr, max_samples=args.max_samples,
                 eval_every=args.eval_every, workers=args.workers, phone_weight=args.phone_weight,
                 patience=args.patience, weight_decay=args.weight_decay, align_weight=args.align_weight,
                 device="cuda" if torch.cuda.is_available() else "cpu")
    best = torch.load(os.path.join(args.out, "best.pt"), weights_only=False)
    log_run("train", os.path.basename(os.path.normpath(args.out)), out=args.out, config=vars(args),
            params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            train_utterances=len(train_utts), best_epoch=best["epoch"],
            dev_wer=best["dev"]["wer"], dev_cer=best["dev"]["cer"],
            final_loss=hist[-1]["loss"], hours=round((time.time() - t0) / 3600, 2),
            notes=args.notes)


if __name__ == "__main__":
    main()
