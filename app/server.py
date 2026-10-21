"""Afono Nullpoint: the desktop app for the Track, sEMG and DOM models.

    python -m app.server            # then open http://127.0.0.1:8765
    Afono Nullpoint.exe             # the packaged app opens its own window

Binds to 127.0.0.1 only. Every request that changes something must carry the header
X-Afono-Client, which web pages on other sites cannot send, so they cannot drive it.
"""
from . import paths

paths.setup()

import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from collections import Counter, deque  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from semg.data import Utterance, load_index  # noqa: E402
from semg.decode import Lexicon, greedy, lexicon_beam_search  # noqa: E402
from semg.page import gaze_word_weights  # noqa: E402
from semg.model import load_model  # noqa: E402
from semg.registry import log_run, read_runs  # noqa: E402
from semg.signal import preprocess  # noqa: E402
from semg.text import normalize_text, wer  # noqa: E402
from semg.user import Profile, _SAFE  # noqa: E402

VERSION = "0.1.0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = os.path.join(paths.DATA, "data", "cache")
INDEX = os.path.join(paths.DATA, "data", "index.json")

app = FastAPI(title="Afono Nullpoint")
app.mount("/assets", StaticFiles(directory=paths.res("site", "assets")), name="assets")
app.mount("/static", StaticFiles(directory=paths.res("app", "static")), name="static")
app.mount("/vendor", StaticFiles(directory=paths.res("app", "static", "vendor")), name="vendor")


@app.middleware("http")
async def local_only(request: Request, call_next):
    """Reject cross-site requests: mutating calls need our header; browsers only let our own pages send it."""
    path = request.url.path
    origin = request.headers.get("origin", "")
    if origin and not (origin.startswith("http://127.0.0.1") or origin.startswith("http://localhost")
                       or origin.startswith("chrome-extension://") or origin.startswith("extension://")):
        return JSONResponse({"detail": "Requests from other sites are not allowed."}, status_code=403)
    needs_header = path.startswith("/api/") and (request.method != "GET" or path in ("/api/commands", "/api/context"))
    if needs_header and not request.headers.get("x-afono-client"):
        return JSONResponse({"detail": "Missing X-Afono-Client header."}, status_code=403)
    response = await call_next(request)
    if not path.startswith("/api/"):
        # The app window must always run the scripts on disk: a cached copy once ran an old self-test.
        response.headers["Cache-Control"] = "no-cache"
    return response


# ------------------------------------------------------------------ recordings to browse
STATE = {"ckpt": None, "model": None, "lock": threading.Lock()}
SAMPLES, SAMPLE_SETS = {}, {}


def load_samples():
    """Full dataset when present (development); otherwise the bundled demo pack."""
    if os.path.exists(INDEX):
        for u in load_index(INDEX):
            if not u.silent or u.split not in ("dev", "test"):
                continue
            key = ("closed-" if u.closed_vocab else "open-") + u.split
            SAMPLES[u.uid] = u
            SAMPLE_SETS.setdefault(key, []).append(u.uid)
        return "dataset"
    pack = paths.res("app", "demo_pack")
    out = os.path.join(paths.DATA, "data", "demo_pack")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(pack, "pack.json"), encoding="utf-8") as f:
        meta = json.load(f)["recordings"]
    arrays = None
    for m in meta:
        prefix = os.path.join(out, m["uid"])
        if not os.path.exists(prefix + "_emg.npy"):
            if arrays is None:
                arrays = np.load(os.path.join(pack, "pack.npz"))
            np.save(prefix + "_emg.npy", arrays[m["uid"]].astype(np.float32) + arrays[m["uid"] + "_offset"])
        closed, split = m["set"].split("-")[0] == "closed", m["set"].split("-")[1]
        u = Utterance(prefix, m["text"], m["text"], True, closed, "demo", 0, split)
        SAMPLES[m["uid"]] = u
        SAMPLE_SETS.setdefault(m["set"], []).append(m["uid"])
    return "demo pack"


SOURCE = load_samples()

# ------------------------------------------------------------------ language model + vocabularies
with open(paths.res("vocab", "lm.json"), encoding="utf-8") as f:
    _LM = json.load(f)
UNI = {k: Counter(v["unigram"]) for k, v in _LM.items()}
BI = {k: Counter({(a, b): c for a, b, c in v["bigram"]}) for k, v in _LM.items()}
LEX = {k: Lexicon(UNI[k].keys(), UNI[k], BI[k]) for k in ("dates", "open")}
with open(paths.res("vocab", "commands.json"), encoding="utf-8") as f:
    COMMANDS = json.load(f)
COMMAND_WORDS = sorted({w for g in COMMANDS["groups"].values() for p in g for w in p.split()})
PERSONAL_WEIGHT = 50


def checkpoints():
    out = [paths.BASE_MODEL.replace("\\", "/")] if os.path.exists(paths.BASE_MODEL) else []
    runs = os.path.join(paths.DATA, "runs")
    for d in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        p = os.path.join(runs, d, "best.pt")
        if os.path.exists(p):
            p = p.replace("\\", "/")
            if p not in out:
                out.append(p)
    return out


def ckpt_name(p):
    return "sEMG base" if p == paths.BASE_MODEL.replace("\\", "/") else os.path.basename(os.path.dirname(p))


def get_model(ckpt=None):
    with STATE["lock"]:
        ckpt = ckpt or STATE["ckpt"] or (checkpoints() or [None])[0]
        if ckpt is None:
            raise HTTPException(503, "No trained sEMG Model found.")
        if ckpt not in checkpoints():
            raise HTTPException(400, "Unknown model")
        if STATE["ckpt"] != ckpt:
            STATE["model"], _ = load_model(ckpt, DEVICE)
            STATE["ckpt"] = ckpt
        return STATE["model"]


def gaze_weights(max_age=1.5):
    """{word: weight} from the Track Model's latest candidates (empty when gaze is off or stale)."""
    if not GAZE or time.time() - GAZE.get("received", 0) > max_age:
        return {}
    g = {int(c["id"][1:]): c["p"] for c in GAZE.get("candidates", []) if c["id"][:1] == "n" and c["id"][1:].isdigit()}
    return gaze_word_weights(CONTEXT.get("targets"), g)


def lexicon_for(vocab, profile, words, gaze=False):
    personal = profile.word_counts()
    if vocab in ("dates", "open"):
        return LEX[vocab]
    if vocab == "personal":
        counts = Counter(UNI["open"])
        for w, c in personal.items():
            counts[w] += PERSONAL_WEIGHT * c
        return Lexicon(counts.keys(), counts, BI["open"])
    if vocab in ("custom", "page"):
        ws = [normalize_text(w) for w in (words or "").replace(",", " ").split()]
        if vocab == "page":
            # The extension's list already holds the command and number words that apply to this page;
            # the full command set (spelling alphabet included) is only a fallback when no page is connected.
            ws += CONTEXT.get("vocabulary") or COMMAND_WORDS
        ws = [w for w in ws if w and " " not in w]
        if not ws:
            raise HTTPException(400, "Type the words that are on screen, separated by spaces.")
        # Lexicon's prior is add-one over counts, so a gaze weight m scales (count + 1).
        weights = gaze_weights() if gaze and vocab == "page" else {}
        counts = Counter({w: (2 + PERSONAL_WEIGHT * personal.get(w, 0)) * weights.get(w, 1.0) - 1 for w in ws})
        return Lexicon(counts.keys(), counts)
    raise HTTPException(400, "Unknown vocabulary")


def preview(x, cols=600):
    T, C = x.shape
    edges = np.linspace(0, T, cols + 1).astype(int)
    env = np.zeros((C, cols, 2), dtype=np.float32)
    for i in range(cols):
        seg = x[edges[i]:max(edges[i] + 1, edges[i + 1])]
        env[:, i, 0], env[:, i, 1] = seg.min(0), seg.max(0)
    return np.round(np.clip(env, -6, 6), 2).tolist()


def signal_for(src, rid, profile):
    if src == "dataset":
        u = SAMPLES.get(rid)
        if u is None:
            raise HTTPException(404, "Unknown recording")
        cached = os.path.join(CACHE, u.uid + ".npy")
        x = np.load(cached).astype(np.float32) if os.path.exists(cached) else preprocess(np.load(u.path + "_emg.npy"))
        return x, u.text
    if src == "recording":
        if not profile.has(rid):
            raise HTTPException(404, "Unknown recording")
        x = preprocess(profile.load_raw(rid), mains_hz=profile.settings()["mains_hz"])
        return x, profile.labels().get(rid, {}).get("text")
    raise HTTPException(400, "source must be 'dataset' or 'recording'")


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return FileResponse(paths.res("app", "static", "index.html"))


LAST_BEAT = [0.0]


@app.get("/api/ping")
def ping():
    return {"ok": True, "app": "Afono Nullpoint", "version": VERSION}


@app.post("/api/heartbeat")
def heartbeat():
    """The app window calls this every few seconds; the desktop launcher exits when it stops."""
    LAST_BEAT[0] = time.time()
    return {"ok": True}


@app.get("/api/status")
def status(profile: str = "default"):
    p = Profile(profile)
    root = os.path.dirname(p.dir)
    profiles = sorted(d for d in os.listdir(root) if _SAFE.match(d)) if os.path.isdir(root) else []
    cur = STATE["ckpt"] or (checkpoints() or [None])[0]
    return {"version": VERSION, "device": str(DEVICE), "ckpt": cur,
            "checkpoints": [{"path": c, "name": ckpt_name(c)} for c in checkpoints()],
            "profile": p.summary(), "settings": p.settings(), "profiles": profiles, "source": SOURCE,
            "sets": {k: len(v) for k, v in SAMPLE_SETS.items()}, "runs": list(reversed(read_runs()))[:20],
            "jobs": [job_view(j) for j in JOBS.values()], "data_dir": paths.DATA,
            "extension_dir": paths.res("extension"), "context": {k: CONTEXT.get(k) for k in ("url", "title", "time")}}


@app.get("/api/samples")
def samples(set: str = "closed-test", limit: int = 80):
    if set not in SAMPLE_SETS:
        return []
    ids = sorted(SAMPLE_SETS[set], key=lambda i: SAMPLES[i].path)[:limit]
    return [{"id": i, "text": SAMPLES[i].text} for i in ids]


@app.get("/api/demo/raw")
def demo_raw(id: str, channels: int = 8):
    """A dataset recording as sensor-like integers, for the simulated sensor."""
    u = SAMPLES.get(id)
    if u is None:
        raise HTTPException(404, "Unknown recording")
    x = np.load(u.path + "_emg.npy")[:, :max(1, min(8, channels))]
    return {"rate": 1000, "rows": np.round(x - np.median(x, 0) + 2048).astype(int).tolist(), "text": u.text}


# ------------------------------------------------------------------ recordings and labels
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
        else:
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
    x = np.asarray(body.get("samples", []), dtype=np.float32)
    try:
        rid = Profile(body.get("profile", "default")).add_recording(x)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": rid, "samples": int(x.shape[0]), "channels": int(x.shape[1]), "source": body.get("source", "sensor")}


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
    """Copy labelled demo recordings (dev split only) into a profile, to try calibration."""
    p = Profile(body.get("profile", "default"))
    which = body.get("set", "closed-dev")
    if not which.endswith("-dev"):
        raise HTTPException(400, "Only dev recordings can be imported; test recordings stay untouched.")
    n = 0
    for uid in SAMPLE_SETS.get(which, [])[: int(body.get("limit", 200))]:
        u, rid = SAMPLES[uid], "ds" + uid[:10]
        if not p.has(rid):
            p.add_recording(np.load(u.path + "_emg.npy"), rid)
            p.add_label(rid, u.text, source="prompted")
            n += 1
    return {"imported": n, "profile": p.summary()}


@app.get("/api/settings")
def get_settings(profile: str = "default"):
    return Profile(profile).settings()


@app.post("/api/settings")
def set_settings(body: dict = Body(...)):
    try:
        return Profile(body.get("profile", "default")).save_settings(**{k: v for k, v in body.items() if k != "profile"})
    except ValueError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ decoding
@app.post("/api/decode")
def decode(body: dict = Body(...)):
    profile = Profile(body.get("profile", "default"))
    x, ref = signal_for(body.get("source"), str(body.get("id", "")), profile)
    model = get_model(body.get("ckpt"))
    if x.shape[1] != model.config["n_channels"]:
        raise HTTPException(400, f"This model expects {model.config['n_channels']} channels and the recording has "
                                 f"{x.shape[1]}. Label a few recordings and run Calibrate to adapt it.")
    vocab = body.get("vocab", "open")
    use_gaze = bool(body.get("gaze", True)) and vocab == "page" and bool(gaze_weights())
    lex = lexicon_for(vocab, profile, body.get("words"), gaze=use_gaze)
    t0 = time.perf_counter()
    with torch.no_grad():
        lp = model(torch.from_numpy(x)[None].to(DEVICE))[0][0].float().cpu().numpy()
    t1 = time.perf_counter()
    hyp = lexicon_beam_search(lp, lex, beam=32, lm_weight=1.0, word_bonus=3.0)
    t2 = time.perf_counter()
    out = {"greedy": greedy(lp), "text": hyp, "reference": ref, "lexicon_size": len(lex), "vocab": vocab,
           "model_ms": round((t1 - t0) * 1000), "decode_ms": round((t2 - t1) * 1000),
           "seconds": round(x.shape[0] / 1000, 2), "preview": preview(x), "gaze_used": use_gaze}
    if ref:
        out["word_accuracy"] = max(0.0, 1 - wer([ref], [hyp]))
    if body.get("send") and hyp:
        hyp = aim(hyp)
        COMMAND_QUEUE.append(hyp)
        out["sent"] = True
    return out


# ------------------------------------------------------------------ DOM Model bridge (browser extension)
CONTEXT, COMMAND_QUEUE, RAN = {}, deque(maxlen=20), deque(maxlen=50)


@app.post("/api/context")
def set_context(body: dict = Body(...)):
    def rect(r):
        try:
            return {k: float(r[k]) for k in ("left", "top", "width", "height")}
        except (KeyError, TypeError, ValueError):
            return None

    targets = [{"n": int(t.get("n", 0)), "label": str(t.get("label", ""))[:60], "kind": str(t.get("kind", ""))[:20],
                "rect": rect(t.get("rect"))}
               for t in body.get("targets", [])[:150]]
    CONTEXT.update({"url": str(body.get("url", ""))[:300], "title": str(body.get("title", ""))[:200], "targets": targets,
                    "vocabulary": [normalize_text(w) for w in body.get("vocabulary", [])[:400] if normalize_text(w)],
                    "time": time.strftime("%H:%M:%S")})
    return {"ok": True}


@app.get("/api/context")
def get_context():
    return CONTEXT or {"targets": [], "vocabulary": []}


@app.get("/api/commands")
def pop_commands():
    out = list(COMMAND_QUEUE)
    COMMAND_QUEUE.clear()
    return {"commands": out}


AIMED = {"click", "double", "right", "double click", "right click"}


def aim(text, max_age=1.5):
    """A bare 'click' / 'double' / 'right' acts on what the Track Model highlights: 'click' -> 'click 25'.
    Without a fresh highlight the command is sent as it is and the extension reports that it has no target."""
    h = GAZE.get("highlight") if GAZE and time.time() - GAZE.get("received", 0) <= max_age else None
    if text in AIMED and isinstance(h, str) and h[:1] == "n" and h[1:].isdigit():
        return f"{text} {h[1:]}"
    return text


@app.post("/api/commands")
def push_command(body: dict = Body(...)):
    text = normalize_text(str(body.get("text", "")))
    if not text:
        raise HTTPException(400, "Empty command")
    text = aim(text)
    COMMAND_QUEUE.append(text)
    return {"queued": text}


@app.post("/api/ran")
def ran(body: dict = Body(...)):
    RAN.appendleft({"text": str(body.get("text", ""))[:120], "action": str(body.get("action", ""))[:20],
                    "time": time.strftime("%H:%M:%S")})
    if RUN.get("phase") == "active":
        RUN["actions"].append({"text": str(body.get("text", ""))[:120], "action": str(body.get("action", ""))[:20],
                               "t": round(time.time() - RUN["t0"], 2)})
    return {"ok": True}


@app.get("/api/ran")
def ran_log():
    return list(RAN)


@app.post("/api/open-extension")
def open_extension():
    folder = paths.res("extension")
    if hasattr(os, "startfile"):
        os.startfile(folder)  # noqa: S606 - opens a local folder in Explorer
    return {"folder": folder}


# ------------------------------------------------------------------ Track Model results
@app.post("/api/track/result")
def track_result(body: dict = Body(...)):
    kind = body.get("kind")
    if kind not in ("validation", "target_test"):
        raise HTTPException(400, "kind must be validation or target_test")
    keep = {k: body[k] for k in ("accuracy_px", "precision_px", "accuracy_deg", "precision_deg", "points", "hit_rate",
                                 "trials", "mean_time_ms", "screen_px", "screen_cm", "distance_cm", "calibration_samples",
                                 "sigma_px", "sigma_deg", "mapping", "lopo_deg", "dropped_points", "pixel_hit_rate",
                                 "centre_hit_rate", "accuracy_deg_poly2", "accuracy_deg_v0", "accuracy_px_poly2",
                                 "accuracy_px_v0", "frames", "face_rate", "eyes_rate", "skipped", "blink_threshold")
            if k in body}
    rec = log_run("evaluate", "track-1", model="track", benchmark=f"Track {kind}", metrics=keep,
                  profile=body.get("profile", "default"), notes=str(body.get("notes", ""))[:200])
    return {"logged": rec["time"]}


# Latest gaze candidates from the Track tab (top-k elements with probabilities), for fusion with the voice decoder.
GAZE = {}


@app.post("/api/track/candidates")
def set_gaze_candidates(body: dict = Body(...)):
    cands = [{"id": str(c.get("id", ""))[:40], "label": str(c.get("label", ""))[:60], "p": float(c.get("p", 0))}
             for c in body.get("candidates", [])[:10]]
    GAZE.clear()
    GAZE.update({"candidates": cands, "p_none": float(body.get("p_none", 1)), "highlight": body.get("highlight"),
                 "fixating": bool(body.get("fixating", False)), "sigma_px": float(body.get("sigma_px", 0)),
                 "gaze": body.get("gaze"), "received": time.time()})
    return {"ok": True}


@app.get("/api/track/candidates")
def get_gaze_candidates():
    if not GAZE:
        return {"candidates": [], "p_none": 1.0, "age_s": None}
    return {**GAZE, "age_s": round(time.time() - GAZE["received"], 3)}


# ------------------------------------------------------------------ Whisper mode (quiet speech, ordinary microphone)
VOICE = {"loading": False, "error": None}


def _voice():
    try:
        from semg import voice
        return voice
    except ImportError as e:  # transformers missing
        raise HTTPException(503, f"Whisper mode needs the transformers package ({e}).")


def _load_voice():
    try:
        _voice().get_decoder()
    except Exception as e:  # noqa: BLE001 - shown in the UI
        VOICE["error"] = str(e)[:300]
    finally:
        VOICE["loading"] = False


@app.get("/api/voice/status")
def voice_status(load: int = 0):
    v = _voice()
    ready = v._DECODER is not None
    if load and not ready and not VOICE["loading"]:
        VOICE.update(loading=True, error=None)
        threading.Thread(target=_load_voice, daemon=True).start()
    cands = v.candidates(CONTEXT.get("targets"))
    return {"ready": ready, "loading": VOICE["loading"], "error": VOICE["error"],
            "model": os.path.basename(v.MODEL_ID.rstrip("/\\")), "bundled": os.path.isdir(v.MODEL_ID),
            "device": v._DECODER.device.type if ready else None, "n_candidates": len(cands),
            "page": CONTEXT.get("title") or None, "candidates": [c["say"] for c in cands[:200]]}


@app.get("/api/voice/prompts")
def voice_prompts(n: int = 20):
    """Self-test prompts for the connected page, weighted like real use (semg.voice.selftest_prompts)."""
    v = _voice()
    return {"prompts": v.selftest_prompts(CONTEXT.get("targets"), n=max(1, min(n, 50))), "set": v.PROMPT_SET}


@app.post("/api/voice/decode")
async def voice_decode(request: Request, send: int = 0, gaze: int = 1, keep: int = 0, label: str = "",
                       profile: str = "default"):
    """Body: 16 kHz mono int16 PCM. Returns the most likely on-screen phrase; queues it for the extension
    when accepted and send=1. With keep=1 and a label (self-test prompts only), the clip is saved to the
    profile folder on this computer for later training."""
    v = _voice()
    raw = await request.body()
    if len(raw) < 3200:
        raise HTTPException(400, "Clip too short.")
    audio = np.frombuffer(raw[: 2 * int(v.MAX_SECONDS * v.SR)], dtype=np.int16).astype(np.float32) / 32768.0
    g = None
    if gaze and GAZE and time.time() - GAZE.get("received", 0) < 1.5:
        g = {int(c["id"][1:]): c["p"] for c in GAZE.get("candidates", []) if c["id"][:1] == "n" and c["id"][1:].isdigit()}
    cands = v.candidates(CONTEXT.get("targets"), commands=v.COMMAND_PHRASES if g else v.COMMANDS_NO_GAZE)
    if label and normalize_text(label) not in {c["say"] for c in cands}:
        cands.append({"say": normalize_text(label), "send": normalize_text(label), "target": None})
    study_words = _study_words() if not label else ()
    for w in study_words:
        if w not in {c["say"] for c in cands}:
            cands.append({"say": w, "send": w, "target": None})
    import asyncio
    r = await asyncio.to_thread(v.get_decoder().decode, audio, cands, g or None)
    if study_words and r["accepted"] and r["say"] in study_words:
        r["study"] = _study_voice(r["say"])        # "start", "done" or "skip" drive the task list, not the page
        return r
    if label:
        # Self-test: right means the same action ("click five four" and "click fifty four" are both badge 54).
        want = {c["say"]: c["send"] for c in cands}.get(normalize_text(label))
        r["label_ok"] = bool(want) and r.get("send") == want
    if send and r["accepted"]:
        r["send"] = aim(r["send"])
        COMMAND_QUEUE.append(r["send"])
        r["queued"] = True
    if keep and label and _SAFE.match(profile):
        d = os.path.join(os.environ.get("AFONO_USER_ROOT", "data/user"), profile, "voice")
        os.makedirs(d, exist_ok=True)
        import wave
        name = time.strftime("%Y%m%d-%H%M%S") + "_" + "-".join(normalize_text(label).split())[:40] + ".wav"
        with wave.open(os.path.join(d, name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(v.SR); w.writeframes(raw)
        r["saved"] = name
    return r


@app.post("/api/voice/result")
def voice_result(body: dict = Body(...)):
    keep = {k: body[k] for k in ("accuracy", "accepted_accuracy", "reject_rate", "trials", "median_candidates",
                                 "median_ms", "style", "mic", "gaze", "prompt_set") if k in body}
    rec = log_run("evaluate", "voice-1", model="voice", benchmark="Whisper mode self-test", metrics=keep,
                  whisper_model=_voice().MODEL_ID, profile=str(body.get("profile", "default"))[:40],
                  notes=str(body.get("notes", ""))[:200])
    return {"logged": rec["time"]}


# ------------------------------------------------------------------ user study: accounts and the guided task list
# Volunteers sign in with a local account (data/study, this laptop only); Nullpoint then runs the task list itself:
# it shows and times each task, checks where the page ended up, and logs the run under the person's pseudonym.
from semg import study as STUDY  # noqa: E402

SESSIONS = {}          # token -> pseudonym, in memory: restarting Nullpoint signs everyone out
RUN = {}               # the task list in progress (one at a time on this laptop)


def _who(request: Request) -> str:
    pid = SESSIONS.get(request.headers.get("x-afono-token", ""))
    if not pid:
        raise HTTPException(401, "Please sign in again.")
    return pid


def _first_name(pid):
    return (STUDY.load_people().get(pid, {}).get("name") or pid).split()[0]


def _session(pid):
    import secrets
    token = secrets.token_urlsafe(24)
    SESSIONS[token] = pid
    return {"token": token, "pid": pid, "first": _first_name(pid)}


@app.post("/api/study/register")
def study_register(body: dict = Body(...)):
    if not body.get("consent"):
        raise HTTPException(400, "Please read and accept the consent statement.")
    if str(body.get("password", "")) != str(body.get("password2", body.get("password", ""))):
        raise HTTPException(400, "The two passwords are not the same.")
    try:
        pid = STUDY.register(body.get("full_name", ""), body.get("username", ""), str(body.get("password", "")),
                             bool(body.get("keep_voice_clips")), str(body.get("group", "prefer-not-to-say")))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _session(pid)


@app.post("/api/study/login")
def study_login(body: dict = Body(...)):
    pid = STUDY.check_login(body.get("username", ""), str(body.get("password", "")))
    if not pid:
        raise HTTPException(401, "Username or password is not right.")
    return _session(pid)


@app.post("/api/study/logout")
def study_logout(request: Request):
    SESSIONS.pop(request.headers.get("x-afono-token", ""), None)
    if RUN and RUN.get("phase") != "finished":
        RUN.clear()
    return {"ok": True}


def _run_view():
    if not RUN:
        return None
    i = RUN["i"]
    return {"mode": RUN["mode"], "phase": RUN["phase"], "i": i, "n": len(STUDY.TASKS),
            "task": STUDY.TASKS[i] if i < len(STUDY.TASKS) else None,
            "elapsed": round(time.time() - RUN["t0"], 1) if RUN["phase"] == "active" else None,
            "rows": [{k: r[k] for k in ("task", "text", "seconds", "result", "verified")} for r in RUN["rows"]],
            "metrics": RUN.get("metrics")}


@app.get("/api/study/me")
def study_me(request: Request):
    pid = _who(request)
    p = STUDY.load_people().get(pid, {})
    return {"pid": pid, "first": _first_name(pid), "keep_voice_clips": p.get("keep_voice_clips", False),
            "progress": STUDY.progress(pid, read_runs()), "tasks": STUDY.TASKS, "start_page": STUDY.START_PAGES[0],
            "sus": STUDY.SUS, "open": STUDY.OPEN, "run": _run_view() if RUN.get("pid") == pid else None,
            "page": {"title": CONTEXT.get("title"), "url": CONTEXT.get("url"), "targets": len(CONTEXT.get("targets", []))}}


@app.post("/api/study/run/start")
def study_run_start(request: Request, body: dict = Body(...)):
    pid = _who(request)
    mode = str(body.get("mode", ""))
    if mode not in STUDY.MODES:
        raise HTTPException(400, "Unknown mode.")
    RUN.clear()
    RUN.update(pid=pid, mode=mode, device=str(body.get("device", ""))[:60], style=str(body.get("style", ""))[:10],
               i=0, phase="ready", rows=[], actions=[], t0=None)
    return _run_view()


def _begin():
    if RUN.get("phase") != "ready":
        return _run_view()
    RUN.update(phase="active", t0=time.time(), actions=[])
    return _run_view()


def _finish(result):
    if RUN.get("phase") != "active":
        return _run_view()
    i, secs, url = RUN["i"], round(time.time() - RUN["t0"], 1), CONTEXT.get("url", "")
    acts = list(RUN["actions"])
    row = {"task": i + 1, "text": STUDY.TASKS[i], "seconds": secs, "result": result,
           "verified": STUDY.check_task(i, url, acts) if result == "done" else False,
           "errors": None, "undos": STUDY.count_undos(i, acts), "actions": [a["text"] for a in acts], "url_end": url}
    RUN["rows"].append(row)

    def recheck():                       # the page may still be loading when "done" is said
        if row["result"] == "done" and row["verified"] is not True:
            row["verified"] = STUDY.check_task(i, CONTEXT.get("url", ""), acts)
            row["url_end"] = CONTEXT.get("url", "")
    threading.Timer(1.5, recheck).start()
    RUN["i"] += 1
    if RUN["i"] >= len(STUDY.TASKS):
        RUN["phase"] = "saving"

        def save():
            recheck()
            _, m = STUDY.save_task_run(RUN["pid"], RUN["mode"], RUN["rows"], RUN["device"], RUN["style"],
                                       "", "nullpoint")
            RUN.update(phase="finished", metrics=m)
        threading.Timer(1.6, save).start()
    else:
        RUN["phase"] = "ready"
    return _run_view()


def _study_words():
    phase = RUN.get("phase")
    return ("start",) if phase == "ready" else ("done", "skip") if phase == "active" else ()


def _study_voice(word):
    return _begin() if word == "start" else _finish("done" if word == "done" else "skipped")


@app.post("/api/study/run/begin")
def study_run_begin(request: Request):
    _who(request)
    return _begin()


@app.post("/api/study/run/finish")
def study_run_finish(request: Request, body: dict = Body(...)):
    _who(request)
    result = str(body.get("result", "done"))
    return _finish(result if result in ("done", "failed", "skipped") else "done")


@app.post("/api/study/run/cancel")
def study_run_cancel(request: Request):
    _who(request)
    RUN.clear()
    return {"ok": True}


@app.get("/api/study/run")
def study_run(request: Request):
    pid = _who(request)
    return _run_view() if RUN.get("pid") == pid else None


@app.post("/api/study/survey")
def study_survey(request: Request, body: dict = Body(...)):
    pid = _who(request)
    try:
        answers = [int(a) for a in body.get("answers", [])]
        would = int(body.get("would_use", 0))
        if would not in (1, 2, 3, 4, 5):
            raise ValueError
        _, score = STUDY.save_survey(pid, answers, would, body.get("open", {}))
    except (ValueError, TypeError):
        raise HTTPException(400, "Please answer every question from 1 to 5.")
    return {"sus": score}


# ------------------------------------------------------------------ calibration (in-process, works when packaged)
JOBS = {}


def job_view(j):
    tail = ""
    if os.path.exists(j["log"]):
        with open(j["log"], encoding="utf-8", errors="replace") as f:
            tail = "".join(f.readlines()[-14:])
    return {"id": j["id"], "profile": j["profile"], "started": j["started"], "state": j["state"], "log": tail}


def _run_calibration(job, profile, base, epochs):
    from semg.calibrate import calibrate
    fh = open(job["log"], "a", encoding="utf-8")

    def log(x):
        fh.write((json.dumps(x, default=str) if not isinstance(x, str) else x) + "\n")
        fh.flush()
    try:
        replay = None if os.path.exists(INDEX) else [SAMPLES[u] for k, v in SAMPLE_SETS.items() if k.endswith("-dev") for u in v]
        rec = calibrate(profile, base, None, epochs, 1e-4, 1.0, INDEX, CACHE, f"Nullpoint calibration ({profile})",
                        log=log, replay_pool=replay)
        log(f"done: held-out word accuracy {rec['before']['lexicon_word_acc']:.3f} -> {rec['after']['lexicon_word_acc']:.3f}")
        job["state"] = "done"
    except Exception as e:
        log("failed: " + str(e))
        log(traceback.format_exc())
        job["state"] = "failed"
    finally:
        fh.close()


@app.post("/api/calibrate")
def calibrate_now(body: dict = Body(...)):
    profile = body.get("profile", "default")
    Profile(profile)
    if any(j["state"] == "running" for j in JOBS.values()):
        raise HTTPException(409, "A calibration is already running.")
    epochs = max(1, min(60, int(body.get("epochs", 12))))
    base = body.get("base") or STATE["ckpt"] or (checkpoints() or [None])[0]
    if base not in checkpoints():
        raise HTTPException(400, "Unknown base model")
    jid = uuid.uuid4().hex[:8]
    os.makedirs(os.path.join(paths.DATA, "runs", "jobs"), exist_ok=True)
    job = {"id": jid, "profile": profile, "log": os.path.join(paths.DATA, "runs", "jobs", f"{jid}.log"),
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": "running"}
    JOBS[jid] = job
    threading.stack_size(64 * 1024 * 1024)   # CPU autograd overflows the default 1 MB thread stack on Windows
    threading.Thread(target=_run_calibration, args=(job, profile, base, epochs), daemon=True).start()
    return job_view(job)


@app.post("/api/model")
def use_model(body: dict = Body(...)):
    get_model(body.get("ckpt"))
    return {"ckpt": STATE["ckpt"]}


def main():
    import uvicorn
    port = int(os.environ.get("AFONO_PORT", "8765"))
    print(f"Afono Nullpoint on http://127.0.0.1:{port}  (data: {paths.DATA})")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
