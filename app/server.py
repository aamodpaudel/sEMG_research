"""Afono Lab: local app to test the model, correct it, and calibrate it.

    python -m app.server            # then open http://127.0.0.1:8765

Runs only on this computer (binds to 127.0.0.1). Decoding uses PyTorch on the GPU if present.
"""
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter

import numpy as np
import torch
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from semg.data import load_index
from semg.decode import Lexicon, greedy, lexicon_beam_search
from semg.model import load_model
from semg.registry import read_runs
from semg.signal import preprocess
from semg.text import normalize_text, wer
from semg.user import Profile, _SAFE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
INDEX, CACHE = "data/index.json", "data/cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Afono Lab")
app.mount("/assets", StaticFiles(directory="site/assets"), name="assets")

STATE = {"ckpt": None, "model": None, "lock": threading.Lock()}
UTTS = {u.uid: u for u in load_index(INDEX)} if os.path.exists(INDEX) else {}
TRAIN = [u for u in UTTS.values() if u.split == "train"]
DATES = Counter(w for u in TRAIN if u.closed_vocab for w in u.text.split())
OPEN = Counter(w for u in TRAIN if not u.closed_vocab for w in u.text.split())
LEX_CACHE = {"dates": Lexicon(DATES.keys(), DATES) if DATES else None,
             "open": Lexicon(OPEN.keys(), OPEN) if OPEN else None}
JOBS = {}
PERSONAL_WEIGHT = 50  # one confirmed use of a word counts like 50 sightings in the book data


def checkpoints():
    out = []
    for d in sorted(os.listdir("runs")) if os.path.isdir("runs") else []:
        p = os.path.join("runs", d, "best.pt")
        if os.path.exists(p):
            out.append(p.replace("\\", "/"))
    return out


def get_model(ckpt=None):
    with STATE["lock"]:
        ckpt = ckpt or STATE["ckpt"] or ("runs/base/best.pt" if os.path.exists("runs/base/best.pt") else None)
        if ckpt is None:
            raise HTTPException(503, "No trained model found. Run python -m semg.train first.")
        if ckpt not in checkpoints():
            raise HTTPException(400, "Unknown checkpoint")
        if STATE["ckpt"] != ckpt:
            STATE["model"], _ = load_model(ckpt, DEVICE)
            STATE["ckpt"] = ckpt
        return STATE["model"]


def lexicon_for(vocab, profile, words):
    if vocab in ("dates", "open"):
        if LEX_CACHE[vocab] is None:
            raise HTTPException(503, "Dataset index missing, so the built-in vocabularies are unavailable.")
        return LEX_CACHE[vocab]
    personal = profile.word_counts()
    if vocab == "personal":
        counts = Counter(OPEN)
        for w, c in personal.items():
            counts[w] += PERSONAL_WEIGHT * c
        return Lexicon(counts.keys(), counts)
    if vocab == "custom":
        ws = [normalize_text(w) for w in (words or "").replace(",", " ").split()]
        ws = [w for w in ws if w and " " not in w]
        if not ws:
            raise HTTPException(400, "Type the words that are on screen, separated by spaces.")
        counts = Counter({w: 1 + PERSONAL_WEIGHT * personal.get(w, 0) for w in ws})
        return Lexicon(counts.keys(), counts)
    raise HTTPException(400, "Unknown vocabulary")


def preview(x, cols=600):
    """Min/max envelope per channel for drawing: (C, cols, 2)."""
    T, C = x.shape
    edges = np.linspace(0, T, cols + 1).astype(int)
    env = np.zeros((C, cols, 2), dtype=np.float32)
    for i in range(cols):
        seg = x[edges[i]:max(edges[i] + 1, edges[i + 1])]
        env[:, i, 0], env[:, i, 1] = seg.min(0), seg.max(0)
    return np.round(np.clip(env, -6, 6), 2).tolist()


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "app", "static", "index.html"))


@app.get("/api/status")
def status(profile: str = "default"):
    p = Profile(profile)
    root = os.path.dirname(p.dir)
    profiles = sorted(d for d in os.listdir(root) if _SAFE.match(d)) if os.path.isdir(root) else []
    return {"device": str(DEVICE), "ckpt": STATE["ckpt"] or "runs/base/best.pt", "checkpoints": checkpoints(),
            "profile": p.summary(), "profiles": profiles, "runs": list(reversed(read_runs()))[:15],
            "dataset": bool(UTTS), "jobs": [job_view(j) for j in JOBS.values()]}


@app.get("/api/samples")
def samples(set: str = "closed-test", limit: int = 60):
    closed, split = {"closed-test": (True, "test"), "open-test": (False, "test"),
                     "closed-dev": (True, "dev"), "open-dev": (False, "dev")}.get(set, (None, None))
    if split is None:
        raise HTTPException(400, "Unknown set")
    pick = [u for u in UTTS.values() if u.split == split and u.silent and u.closed_vocab == closed]
    pick.sort(key=lambda u: u.path)
    return [{"id": u.uid, "text": u.text} for u in pick[:limit]]


@app.get("/api/recordings")
def recordings(profile: str = "default"):
    p = Profile(profile)
    labels = p.labels()
    return [{"id": r, "label": labels.get(r, {}).get("text")} for r in reversed(p.recordings())]


@app.post("/api/recordings")
async def upload(profile: str = "default", file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 50_000_000:
        raise HTTPException(413, "File too large (max 50 MB)")
    name = (file.filename or "").lower()
    try:
        if name.endswith(".npy"):
            x = np.load(io.BytesIO(raw), allow_pickle=False)
        else:  # CSV / TSV: one row per sample, one column per channel
            x = np.loadtxt(io.StringIO(raw.decode("utf-8", "replace")), delimiter="," if b"," in raw[:2000] else None)
    except Exception as e:
        raise HTTPException(400, f"Could not read the recording: {e}")
    try:
        rid = Profile(profile).add_recording(np.asarray(x, dtype=np.float32))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": rid, "samples": int(x.shape[0]), "channels": int(x.shape[1])}


@app.post("/api/recordings/stream")
def upload_stream(body: dict = Body(...)):
    """Samples captured in the browser over Web Serial: {"profile", "samples": [[c1..cN], ...]}."""
    x = np.asarray(body.get("samples", []), dtype=np.float32)
    try:
        rid = Profile(body.get("profile", "default")).add_recording(x)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": rid, "samples": int(x.shape[0]), "channels": int(x.shape[1])}


@app.post("/api/decode")
def decode(body: dict = Body(...)):
    profile = Profile(body.get("profile", "default"))
    src, rid = body.get("source"), str(body.get("id", ""))
    if src == "dataset":
        u = UTTS.get(rid)
        if u is None:
            raise HTTPException(404, "Unknown dataset recording")
        x, ref = np.load(os.path.join(CACHE, u.uid + ".npy")).astype(np.float32), u.text
    elif src == "recording":
        if not profile.has(rid):
            raise HTTPException(404, "Unknown recording")
        x, ref = preprocess(profile.load_raw(rid)), profile.labels().get(rid, {}).get("text")
    else:
        raise HTTPException(400, "source must be 'dataset' or 'recording'")
    model = get_model(body.get("ckpt"))
    if x.shape[1] != model.config["n_channels"]:
        raise HTTPException(400, f"This model expects {model.config['n_channels']} channels and the recording has "
                                 f"{x.shape[1]}. Label a few recordings and run Calibrate to adapt it.")
    lex = lexicon_for(body.get("vocab", "open"), profile, body.get("words"))
    t0 = time.perf_counter()
    with torch.no_grad():
        lp = model(torch.from_numpy(x)[None].to(DEVICE))[0][0].float().cpu().numpy()
    t1 = time.perf_counter()
    hyp = lexicon_beam_search(lp, lex, beam=32, lm_weight=1.0, word_bonus=3.0)
    t2 = time.perf_counter()
    out = {"greedy": greedy(lp), "text": hyp, "reference": ref, "lexicon_size": len(lex),
           "model_ms": round((t1 - t0) * 1000), "decode_ms": round((t2 - t1) * 1000),
           "seconds": round(x.shape[0] / 1000, 2), "preview": preview(x)}
    if ref:
        out["word_accuracy"] = max(0.0, 1 - wer([ref], [hyp]))
    return out


@app.post("/api/labels")
def label(body: dict = Body(...)):
    p = Profile(body.get("profile", "default"))
    text, hyp = body.get("text", ""), body.get("model_output", "")
    source = "confirmed" if normalize_text(text) == normalize_text(hyp) else "correction"
    try:
        rec = p.add_label(str(body.get("id", "")), text, source=source, model_output=hyp)
    except KeyError:
        raise HTTPException(404, "Unknown recording")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"saved": rec, "profile": p.summary()}


@app.post("/api/import")
def import_demo(body: dict = Body(...)):
    from semg.calibrate import import_dataset
    p = Profile(body.get("profile", "default"))
    n = import_dataset(p, INDEX, body.get("set", "closed-dev"), int(body.get("limit", 200)))
    return {"imported": n, "profile": p.summary()}


def job_view(j):
    tail = ""
    if os.path.exists(j["log"]):
        with open(j["log"], encoding="utf-8", errors="replace") as f:
            tail = "".join(f.readlines()[-14:])
    code = j["proc"].poll()
    return {"id": j["id"], "profile": j["profile"], "started": j["started"],
            "state": "running" if code is None else ("done" if code == 0 else "failed"), "log": tail}


@app.post("/api/calibrate")
def calibrate(body: dict = Body(...)):
    profile = body.get("profile", "default")
    Profile(profile)  # validates the name
    if any(j["proc"].poll() is None for j in JOBS.values()):
        raise HTTPException(409, "A calibration is already running.")
    epochs = max(1, min(60, int(body.get("epochs", 12))))
    base = body.get("base") or STATE["ckpt"] or "runs/base/best.pt"
    if base not in checkpoints():
        raise HTTPException(400, "Unknown base checkpoint")
    jid = uuid.uuid4().hex[:8]
    os.makedirs("runs/jobs", exist_ok=True)
    logp = f"runs/jobs/{jid}.log"
    cmd = [sys.executable, "-u", "-m", "semg.calibrate", "--profile", profile, "--base", base,
           "--epochs", str(epochs), "--notes", f"Afono Lab calibration ({profile})"]
    fh = open(logp, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    JOBS[jid] = {"id": jid, "profile": profile, "proc": proc, "log": logp,
                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    return job_view(JOBS[jid])


@app.post("/api/model")
def use_model(body: dict = Body(...)):
    get_model(body.get("ckpt"))
    return {"ckpt": STATE["ckpt"]}


def main():
    import uvicorn
    port = int(os.environ.get("AFONO_PORT", "8765"))
    print(f"Afono Lab on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
