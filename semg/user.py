"""Per-user data: recordings, labels (corrections) and a personal word prior.

Layout under data/user/<profile>/:
    recordings/<rid>_emg.npy   raw EMG, (T, C) at 1000 Hz
    labels.jsonl               {"rid", "text", "source", "model_output", "time"}
The personal prior is rebuilt from labels.jsonl, so every confirmed or corrected
sentence immediately changes what the decoder prefers (no training needed).
"""
import datetime as _dt
import hashlib
import json
import os
import re
import uuid
from collections import Counter

import numpy as np

from .data import Utterance
from .text import normalize_text

ROOT = os.environ.get("AFONO_USER_ROOT", "data/user")
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class Profile:
    def __init__(self, name: str = "default", root: str = ROOT):
        if not _SAFE.match(name):
            raise ValueError("profile name must be letters, digits, _ or - (max 40)")
        self.name = name
        self.dir = os.path.join(root, name)
        self.rec_dir = os.path.join(self.dir, "recordings")
        os.makedirs(self.rec_dir, exist_ok=True)
        self.labels_path = os.path.join(self.dir, "labels.jsonl")

    # settings ---------------------------------------------------------
    DEFAULTS = {"mains_hz": 60}

    def settings(self) -> dict:
        path = os.path.join(self.dir, "settings.json")
        out = dict(self.DEFAULTS)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out.update(json.load(f))
        return out

    def save_settings(self, **kw) -> dict:
        cur = self.settings()
        if "mains_hz" in kw:
            if int(kw["mains_hz"]) not in (50, 60):
                raise ValueError("mains_hz must be 50 or 60")
            cur["mains_hz"] = int(kw["mains_hz"])
        with open(os.path.join(self.dir, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(cur, f)
        return cur

    # recordings -------------------------------------------------------
    def add_recording(self, emg: np.ndarray, rid: str | None = None) -> str:
        emg = np.asarray(emg, dtype=np.float32)
        if emg.ndim != 2 or emg.shape[0] < 200 or emg.shape[1] > 64:
            raise ValueError(f"expected (T>=200, C<=64) EMG array, got {emg.shape}")
        rid = rid or uuid.uuid4().hex[:12]
        if not _SAFE.match(rid):
            raise ValueError("bad recording id")
        np.save(self.prefix(rid) + "_emg.npy", emg)
        return rid

    def prefix(self, rid: str) -> str:
        return os.path.join(self.rec_dir, rid)

    def has(self, rid: str) -> bool:
        return bool(_SAFE.match(rid)) and os.path.exists(self.prefix(rid) + "_emg.npy")

    def load_raw(self, rid: str) -> np.ndarray:
        return np.load(self.prefix(rid) + "_emg.npy")

    def recordings(self) -> list[str]:
        return sorted(f[:-8] for f in os.listdir(self.rec_dir) if f.endswith("_emg.npy"))

    # labels -----------------------------------------------------------
    def add_label(self, rid: str, text: str, source: str = "correction", model_output: str = "") -> dict:
        if not self.has(rid):
            raise KeyError(rid)
        rec = {"rid": rid, "text": normalize_text(text), "source": source, "model_output": model_output,
               "time": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        if not rec["text"]:
            raise ValueError("label text is empty after normalisation")
        with open(self.labels_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def labels(self) -> dict[str, dict]:
        """Latest label per recording (a later correction overrides an earlier one)."""
        out = {}
        if os.path.exists(self.labels_path):
            with open(self.labels_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out[r["rid"]] = r
        return out

    def word_counts(self) -> Counter:
        return Counter(w for r in self.labels().values() for w in r["text"].split())

    def utterances(self) -> list[Utterance]:
        """Labelled recordings as Utterances; ~20% held out (by id hash) to measure improvement."""
        utts, mains = [], self.settings()["mains_hz"]
        for rid, r in sorted(self.labels().items()):
            frac = int(hashlib.md5(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            utts.append(Utterance(self.prefix(rid), r["text"], r["text"], True, False, f"user:{self.name}", 0,
                                  "dev" if frac < 0.2 else "train", mains))
        return utts

    def summary(self) -> dict:
        labels = self.labels()
        return {"profile": self.name, "recordings": len(self.recordings()), "labelled": len(labels),
                "corrections": sum(1 for r in labels.values() if r["source"] == "correction"),
                "vocabulary": len(self.word_counts())}
