"""Gaddy & Klein (2020) silent-speech EMG dataset: indexing, splitting, caching, batching.

Split rule (matches the official repo): sentences listed in testset_largedev.json are
dev/test *only* for silent recordings; voiced recordings of those sentences are dropped
so the model never hears a test sentence during training.
"""
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .phonemes import encode_phones
from .signal import preprocess, augment, augment_v2
from .text import normalize_text, encode


@dataclass
class Utterance:
    path: str          # prefix, e.g. .../5-4/12  -> 12_emg.npy, 12_info.json
    text: str          # normalised text
    raw_text: str
    silent: bool
    closed_vocab: bool
    book: str
    sentence_index: int
    split: str = "train"
    mains_hz: int = 60

    @property
    def uid(self) -> str:
        key = self.path if self.mains_hz == 60 else f"{self.path}@{self.mains_hz}"
        return hashlib.md5(key.encode()).hexdigest()[:16]


def _hash_frac(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def index_dataset(root: str, testset_file: str) -> list[Utterance]:
    with open(testset_file) as f:
        ts = json.load(f)
    test_keys = {tuple(x) for x in ts["test"]}
    dev_keys = {tuple(x) for x in ts["dev"]}
    utts = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.endswith("_info.json"):
                continue
            prefix = os.path.join(dirpath, fname[: -len("_info.json")])
            if not os.path.exists(prefix + "_emg.npy"):
                continue
            with open(prefix + "_info.json", "rb") as f:
                info = json.loads(f.read().decode("utf-8", errors="replace"))
            if info.get("sentence_index", -1) < 0:  # silence boundary clips
                continue
            text = normalize_text(info.get("text", ""))
            if not text:
                continue
            rel = os.path.relpath(prefix, root).replace("\\", "/")
            silent = "silent" in rel
            closed = "closed_vocab" in rel
            u = Utterance(prefix, text, info["text"], silent, closed,
                          str(info.get("book", "")), int(info["sentence_index"]))
            key = (u.book, u.sentence_index)
            if closed:  # no official split: 10% dev / 10% test of silent, by recording
                r = _hash_frac(rel)
                u.split = "train" if not silent or r >= 0.2 else ("dev" if r < 0.1 else "test")
            elif key in test_keys or key in dev_keys:
                if not silent:
                    continue
                u.split = "test" if key in test_keys else "dev"
            utts.append(u)
    utts.sort(key=lambda u: u.path)
    return utts


def _cache_one(args):
    u_path, out, mains = args
    if not os.path.exists(out):
        x = preprocess(np.load(u_path + "_emg.npy"), mains_hz=mains)
        np.save(out, x.astype(np.float16))
    return out


def build_cache(utts: list[Utterance], cache_dir: str, workers: int = 6):
    os.makedirs(cache_dir, exist_ok=True)
    jobs = [(u.path, os.path.join(cache_dir, u.uid + ".npy"), u.mains_hz) for u in utts]
    todo = [j for j in jobs if not os.path.exists(j[1])]
    if not todo:
        return
    if workers <= 1 or len(todo) < 32:          # also the path used inside the packaged app
        for j in todo:
            _cache_one(j)
        return
    from tqdm import tqdm
    with Pool(workers) as p:
        for _ in tqdm(p.imap_unordered(_cache_one, todo, chunksize=16), total=len(todo), desc="cache"):
            pass


def save_index(utts, path):
    with open(path, "w") as f:
        json.dump([asdict(u) for u in utts], f)


def load_index(path) -> list[Utterance]:
    with open(path) as f:
        return [Utterance(**d) for d in json.load(f)]


class EMGDataset(Dataset):
    def __init__(self, utts: list[Utterance], cache_dir: str, train: bool = False,
                 max_len: int = 30000, seed: int = 0, phones: bool = False, aug: str = "v1",
                 align_dir: str | None = None):
        self.cache_dir, self.train, self.max_len = cache_dir, train, max_len
        self.phones, self.aug, self.align_dir = phones, aug, align_dir
        if align_dir and aug != "v1":
            raise ValueError("aligned voiced targets are only supported with aug v1 (no time stretch)")
        if train:  # never crop training audio: the transcript would no longer match
            utts = [u for u in utts if self._raw_len(u) <= max_len]
        self.utts = list(utts)
        self.rng = np.random.default_rng(seed)
        self.lengths = None

    def __len__(self):
        return len(self.utts)

    def _raw_len(self, u):
        return np.load(os.path.join(self.cache_dir, u.uid + ".npy"), mmap_mode="r").shape[0]

    def load(self, i):
        return np.load(os.path.join(self.cache_dir, self.utts[i].uid + ".npy")).astype(np.float32)

    def __getitem__(self, i):
        u = self.utts[i]
        x = self.load(i)[: self.max_len]
        tgt_path = os.path.join(self.align_dir, u.uid + ".npy") if self.align_dir and u.silent else None
        tgt = np.load(tgt_path).astype(np.float32) if tgt_path and os.path.exists(tgt_path) else None
        if self.train:
            if self.aug == "v1":
                x, shift = augment(x, self.rng, return_shift=True)
                if tgt is not None:
                    tgt = tgt[shift // 8:]
            else:
                x = augment_v2(x, self.rng)
        item = {"emg": torch.from_numpy(x), "label": torch.tensor(encode(u.text)),
                "mode": 0 if u.silent else 1, "text": u.text}
        if self.phones:
            item["phones"] = torch.tensor(encode_phones(u.text))
        if self.align_dir:
            item["align"] = torch.from_numpy(tgt) if tgt is not None else None
        return item

    def compute_lengths(self):
        if self.lengths is None:
            self.lengths = [min(self.max_len, np.load(os.path.join(self.cache_dir, u.uid + ".npy"),
                                                      mmap_mode="r").shape[0]) for u in self.utts]
        return self.lengths


def collate(batch):
    lens = torch.tensor([b["emg"].shape[0] for b in batch])
    emg = torch.zeros(len(batch), int(lens.max()), batch[0]["emg"].shape[1])
    for i, b in enumerate(batch):
        emg[i, : lens[i]] = b["emg"]
    labels = torch.cat([b["label"] for b in batch])
    label_lens = torch.tensor([len(b["label"]) for b in batch])
    out = {"emg": emg, "lengths": lens, "labels": labels, "label_lengths": label_lens,
           "mode": torch.tensor([b["mode"] for b in batch]), "texts": [b["text"] for b in batch]}
    if "align" in batch[0]:
        V = next((b["align"].shape[1] for b in batch if b["align"] is not None), 1)
        T = max([b["align"].shape[0] for b in batch if b["align"] is not None] or [1])
        al = torch.zeros(len(batch), T, V)
        al_len = torch.zeros(len(batch), dtype=torch.long)
        for i, b in enumerate(batch):
            if b["align"] is not None:
                al[i, : len(b["align"])] = b["align"]
                al_len[i] = len(b["align"])
        out["align"], out["align_lengths"] = al, al_len
    if "phones" in batch[0]:
        out["phones"] = torch.cat([b["phones"] for b in batch])
        out["phone_lengths"] = torch.tensor([len(b["phones"]) for b in batch])
    return out


class BucketSampler(Sampler):
    """Batches of similar length, capped at `max_samples` total (padded) time steps."""

    def __init__(self, lengths, max_samples=200_000, shuffle=True, seed=0):
        self.lengths, self.max_samples, self.shuffle = lengths, max_samples, shuffle
        self.rng = np.random.default_rng(seed)

    def _batches(self):
        idx = np.arange(len(self.lengths))
        if self.shuffle:
            self.rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), 256):
            chunk = sorted(idx[s:s + 256], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                m = max(cur_max, self.lengths[i])
                if cur and m * (len(cur) + 1) > self.max_samples:
                    batches.append(cur)
                    cur, m = [], self.lengths[i]
                cur.append(int(i))
                cur_max = m
            if cur:
                batches.append(cur)
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())
