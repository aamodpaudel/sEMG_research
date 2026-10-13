"""Afono Track Model: offline evaluation that needs no human.

    python -m semg.gaze_eval sim            # SIMULATION: snap-to-element vs pixel steering, by gaze error in degrees
    python -m semg.gaze_eval mpii-extract   # run MediaPipe Face Landmarker (CPU) over MPIIFaceGaze, cache landmarks
    python -m semg.gaze_eval mpii           # landmark-feature baselines on MPIIFaceGaze (leave-one-subject-out,
                                            # and 9-point personal calibration per subject)
    python -m semg.gaze_eval mpii-screen    # what Nullpoint ships: features -> screen pixels, 9-point calibration,
                                            # poly2 vs the population head (LOSO), error converted to degrees
    python -m semg.gaze_eval export-pop     # train the population head on all 15 subjects -> app/static/track_pop.json

`sim` is a simulation. Element layouts are real rectangles (the DOM-0 fixture rendered in headless Chrome and
detected by the DOM Model, plus synthetic layouts with typical web target sizes), but the gaze error is drawn
from a model: an isotropic Gaussian scaled so the mean angular error equals 1-5 degrees, optionally with a
per-session constant bias. It quantifies "snapping beats pixel steering" under that error model; it is not a
measurement of any person's gaze. The snapping rule is the same as app/static/track.js (tests/test_track.py
checks that the two implementations agree).
"""
import argparse
import functools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy.special import ndtr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Viewing geometry for the simulation: 1920 x 1080 screen, 34 cm wide, viewed from 60 cm.
SCREEN_W, SCREEN_H, SCREEN_CM, DIST_CM = 1920, 1080, 34.0, 60.0
PX_PER_CM = SCREEN_W / SCREEN_CM


def deg2px(deg, px_per_cm=PX_PER_CM, dist_cm=DIST_CM):
    return math.tan(math.radians(deg)) * dist_cm * px_per_cm


# ----------------------------------------------------------------------------- snapping (mirrors track.js)
def rect_likelihood(x, y, rects, sigma):
    """p(gaze | element): the user looks uniformly inside the rect, gaze error is isotropic N(0, sigma^2).
    x, y: arrays (n,), rects: (m, 4) [left, top, width, height]. Returns (n, m)."""
    x = np.asarray(x, float)[:, None]
    y = np.asarray(y, float)[:, None]
    L, T = rects[None, :, 0], rects[None, :, 1]
    W, H = np.maximum(1, rects[None, :, 2]), np.maximum(1, rects[None, :, 3])
    px = ndtr((L + W - x) / sigma) - ndtr((L - x) / sigma)
    py = ndtr((T + H - y) / sigma) - ndtr((T - y) / sigma)
    return px * py / (W * H)


def posterior(x, y, rects, sigma, p_none=0.05, screen=(SCREEN_W, SCREEN_H)):
    """Posterior over elements (n, m) and the 'none' hypothesis (n,), as in track.js posterior()."""
    sigma = np.asarray(sigma, float).reshape(-1, 1) if np.ndim(sigma) else sigma
    m = len(rects)
    s = (1 - p_none) / m * rect_likelihood(x, y, rects, sigma)
    none = p_none / (screen[0] * screen[1])
    Z = s.sum(1) + none
    return s / Z[:, None], none / Z


def nearest_centre(x, y, rects):
    cx, cy = rects[:, 0] + rects[:, 2] / 2, rects[:, 1] + rects[:, 3] / 2
    return np.argmin((np.asarray(x)[:, None] - cx) ** 2 + (np.asarray(y)[:, None] - cy) ** 2, 1)


def inside(x, y, rects, idx):
    r = rects[idx]
    return (x >= r[:, 0]) & (x <= r[:, 0] + r[:, 2]) & (y >= r[:, 1]) & (y <= r[:, 1] + r[:, 3])


# ----------------------------------------------------------------------------- layouts
def page_layout(page_path, width=SCREEN_W, height=SCREEN_H):
    """Render a page in headless Chrome at 1920x1080 and return the DOM Model's targets in the first viewport
    (label + rect). Consumes extension/dom-model.js read-only."""
    from .dom_eval import find_browser
    ext = os.path.join(ROOT, "extension")
    url = lambda p: "file:///" + p.replace("\\", "/")  # noqa: E731
    page = f"""<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">
<pre id="out">running</pre>
<iframe id="f" src="{url(os.path.abspath(page_path))}" style="position:absolute;left:0;top:0;width:{width}px;height:{height}px;border:0"></iframe>
<script src="{url(os.path.join(ext, 'dom-model.js'))}"></script>
<script>
document.getElementById("f").addEventListener("load", function () {{
  var found = AfonoDOM.detect(this.contentDocument);
  var out = found.map(function (t) {{ var r = t.rect; return {{label: t.label, kind: t.kind, rect: [r.left, r.top, r.width, r.height]}}; }});
  document.getElementById("out").textContent = "RECTS " + JSON.stringify(out);
}});
</script></body></html>"""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "dump.html")
        with open(p, "w", encoding="utf-8") as f:
            f.write(page)
        prof = os.path.join(tmp, "prof")
        out = subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--allow-file-access-from-files",
                              "--host-resolver-rules=MAP * ~NOTFOUND",
                              f"--user-data-dir={prof}", f"--window-size={width},{height}", "--virtual-time-budget=5000",
                              "--dump-dom", url(p)], capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"RECTS (\[.*?\])</pre>", out, re.S)
    if not m:
        raise RuntimeError("fixture page did not report rects")
    items = json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
    return [{"label": t["label"], "rect": t["rect"]} for t in items
            if t["rect"][2] > 0 and t["rect"][3] > 0 and t["rect"][0] >= 0 and t["rect"][1] >= 0
            and t["rect"][0] + t["rect"][2] <= width and t["rect"][1] + t["rect"][3] <= height]


@functools.lru_cache(maxsize=1)
def real_layouts():
    """DOM-0 fixture plus any saved real pages under extension/test/pages/ (first viewport, network blocked)."""
    out = {}
    pages = [("dom0_fixture", os.path.join(ROOT, "extension", "test", "fixture.html"))]
    pdir = os.path.join(ROOT, "extension", "test", "pages")
    if os.path.isdir(pdir):
        pages += [(f[:-5], os.path.join(pdir, f)) for f in sorted(os.listdir(pdir)) if f.endswith(".html")]
    for name, path in pages:
        try:
            items = page_layout(path)
            if len(items) >= 2:
                out[name] = items
        except Exception as e:  # noqa: BLE001
            print(f"layout {name} skipped: {e}")
    return out


WORDS = ("home news sports weather search login account settings help contact about shop cart checkout music video "
         "photos maps mail calendar drive docs sheets slides share download upload print save delete edit copy paste "
         "undo redo open close next previous back forward refresh menu profile logout inbox sent drafts archive spam "
         "reply forward star label filter sort more less zoom play pause stop mute volume").split()


def synthetic_layouts():
    """Layouts with typical target sizes (CSS px at 100% zoom on a 1920x1080 screen)."""
    L = {}
    # 1. Toolbar: 24 icon buttons, 32x32 with 8 px gaps (e.g. a docs editor toolbar).
    L["toolbar_32px"] = [{"label": WORDS[i], "rect": [300 + i * 40, 120, 32, 32]} for i in range(24)]
    # 2. Grid of 44x44 buttons with 8 px gaps (WCAG 2.5.5 minimum target size), 10 x 6.
    L["grid_44px"] = [{"label": WORDS[(r * 10 + c) % len(WORDS)] + ("" if r * 10 + c < len(WORDS) else str(r)),
                       "rect": [700 + c * 52, 380 + r * 52, 44, 44]} for r in range(6) for c in range(10)]
    # 3. Search results: 10 results, title link 560x24, then 3 small 60x18 links under each ("Cached", "Similar",
    #    "Share"); plus a nav bar of 8 links. Repeated labels on purpose: gaze must pick which "share".
    res = [{"label": w, "rect": [180 + 90 * i, 20, 70, 28]} for i, w in enumerate(WORDS[:8])]
    for i in range(10):
        top = 110 + i * 92
        res.append({"label": f"result {WORDS[10 + i]}", "rect": [180, top, 560, 24]})
        for j, w in enumerate(["cached", "similar", "share"]):
            res.append({"label": w, "rect": [180 + j * 72, top + 34, 60, 18]})
    L["search_results"] = res
    # 4. Mail list: 20 rows of 1100x36 (subject lines) with a 28x28 star and a 28x28 archive icon on each row.
    mail = []
    for i in range(20):
        top = 140 + i * 44
        mail.append({"label": "star", "rect": [300, top + 4, 28, 28]})
        mail.append({"label": f"message {WORDS[i]}", "rect": [340, top, 1100, 36]})
        mail.append({"label": "archive", "rect": [1452, top + 4, 28, 28]})
    L["mail_list"] = mail
    # 5. Large grid (the Nullpoint target test): 4 x 3 cells filling the screen with gaps.
    gw, gh = (SCREEN_W * 0.88 - 3 * 58) / 4, (SCREEN_H * 0.88 - 2 * 32) / 3
    L["grid_12_large"] = [{"label": str(i + 1), "rect": [115 + (i % 4) * (gw + 58), 65 + (i // 4) * (gh + 32), gw, gh]}
                          for i in range(12)]
    return L


# ----------------------------------------------------------------------------- simulation
def simulate_layout(items, err_deg, reps, rng, bias_share=0.0, fixation="centre50", voice_acc=0.9, k=3, n_val=5):
    """One layout, one error level. Each element is the target `reps` times.
    Error model: mean angular error = err_deg. Isotropic Gaussian per axis sigma_true = mean_px / sqrt(pi/2).
    bias_share: share of the error variance that is a constant offset per simulated session (one session = one
    pass over all targets); the rest is fresh noise per trial. The snapper's sigma is estimated from a simulated
    5-point validation (n_val points, same error model), as in the app.
    Returns per-trial hit arrays for each method."""
    rects = np.array([t["rect"] for t in items], float)
    labels = [t["label"].lower() for t in items]
    m = len(rects)
    sig = deg2px(err_deg) / math.sqrt(math.pi / 2)
    sb, sn = sig * math.sqrt(bias_share), sig * math.sqrt(1 - bias_share)
    tgt = np.repeat(np.arange(m), reps)
    n = len(tgt)
    # Where on the element the user looks.
    r = rects[tgt]
    if fixation == "centre50":
        fx = r[:, 0] + r[:, 2] * rng.uniform(0.25, 0.75, n)
        fy = r[:, 1] + r[:, 3] * rng.uniform(0.25, 0.75, n)
    else:
        fx = r[:, 0] + r[:, 2] * rng.uniform(0, 1, n)
        fy = r[:, 1] + r[:, 3] * rng.uniform(0, 1, n)
    # Error: session bias (one per repetition pass) + per-trial noise.
    rep_of_trial = np.tile(np.arange(reps), m)
    sess_bias = rng.normal(0, sb, (reps, 2)) if sb > 0 else np.zeros((reps, 2))
    bias = sess_bias[rep_of_trial]
    gx = fx + bias[:, 0] + rng.normal(0, sn, n)
    gy = fy + bias[:, 1] + rng.normal(0, sn, n)
    # sigma estimated per session from a simulated validation: sqrt(mean |e|^2 / 2) over n_val points x 30 frames.
    sig_hat_s = np.empty(reps)
    for s in range(reps):
        e = sess_bias[s] + rng.normal(0, sn, (n_val * 30, 2))
        sig_hat_s[s] = math.sqrt((e ** 2).sum(1).mean() / 2)
    sig_hat = sig_hat_s[rep_of_trial]
    post, pnone = posterior(gx, gy, rects, sig_hat)
    order = np.argsort(-post, 1)
    snap = order[:, 0] == tgt
    pixel = inside(gx, gy, rects, tgt)                    # the cursor is where you look; click lands if inside
    centre = nearest_centre(gx, gy, rects) == tgt         # the v0 rule
    topk = (order[:, :k] == tgt[:, None]).any(1)
    # Voice fusion: the voice decoder outputs the target's label with probability voice_acc, else a uniformly random
    # other word from the page vocabulary (an assumption, not a measured confusion matrix).
    vocab = sorted(set(labels))
    V = len(vocab)
    lab = np.array([vocab.index(w) for w in labels])
    true_w = lab[tgt]
    if V > 1:
        other = rng.integers(0, V - 1, n)
        other = other + (other >= true_w)
        heard = np.where(rng.random(n) < voice_acc, true_w, other)
    else:
        heard = true_w
    p_wrong = (1 - voice_acc) / max(1, V - 1)
    cand = order[:, :k]
    rows = np.arange(n)[:, None]
    pv = np.where(lab[cand] == heard[:, None], voice_acc, p_wrong)
    fused = cand[np.arange(n), np.argmax(post[rows, cand] * pv, 1)] == tgt           # hard top-k, then voice
    pv1 = lab[cand] == true_w[:, None]
    fused_perfect = pv1.any(1) & (cand[np.arange(n), np.argmax(post[rows, cand] * (pv1 + 1e-12), 1)] == tgt)
    pv_all = np.where(lab[None, :] == heard[:, None], voice_acc, p_wrong)               # soft: all elements
    fused_soft = np.argmax(post * pv_all, 1) == tgt
    cnt = np.bincount(lab, minlength=V)[heard]                                        # voice alone: ties by label
    voice_only = (heard == true_w) & (rng.random(n) < 1 / cnt)
    return {"snap": snap, "pixel": pixel, "centre": centre, "topk": topk, "fused": fused,
            "fused_perfect_voice": fused_perfect, "fused_soft": fused_soft, "voice_only": voice_only}


def run_sim(errors=(1, 2, 3, 4, 5), reps=200, seed=1234, bias_share=0.0, fixation="centre50", voice_acc=0.9,
            include_fixture=True, notes="", log=True):
    rng = np.random.default_rng(seed)
    layouts = {}
    if include_fixture:
        layouts.update(real_layouts())
    layouts.update(synthetic_layouts())
    out = {"simulation": True, "screen_px": [SCREEN_W, SCREEN_H], "screen_cm": SCREEN_CM, "distance_cm": DIST_CM,
           "px_per_degree": deg2px(1), "reps_per_element": reps, "seed": seed, "bias_share": bias_share,
           "fixation": fixation, "voice_accuracy_assumed": voice_acc, "layouts": {}, "results": []}
    for name, items in layouts.items():
        rects = np.array([t["rect"] for t in items])
        out["layouts"][name] = {"elements": len(items), "median_w": float(np.median(rects[:, 2])),
                                "median_h": float(np.median(rects[:, 3])),
                                "vocabulary": len({t["label"].lower() for t in items})}
    synth = set(synthetic_layouts())
    groups = {"ALL real pages": [n for n in layouts if n not in synth], "ALL synthetic": [n for n in layouts if n in synth]}
    for err in errors:
        per = {}
        for name, items in layouts.items():
            r = simulate_layout(items, err, reps, rng, bias_share, fixation, voice_acc)
            per[name] = r
            row = {"error_deg": err, "layout": name, "trials": int(len(r["snap"]))}
            for key, v in r.items():
                row[key] = float(v.mean())
            out["results"].append(row)
        # Pooled with equal weight per layout; CIs bootstrap over layouts.
        for gname, names in groups.items():
            if not names:
                continue
            row = {"error_deg": err, "layout": f"{gname} ({len(names)} layouts, equal weight)",
                   "trials": int(sum(len(per[n]["snap"]) for n in names))}
            for key in per[names[0]]:
                arrs = [per[n][key] for n in names]
                row[key] = float(np.mean([a.mean() for a in arrs]))
                row[key + "_ci"] = _pooled_ci(arrs)
            row["snap_minus_pixel"] = row["snap"] - row["pixel"]
            row["snap_minus_pixel_ci"] = _pooled_ci([per[n]["snap"].astype(float) - per[n]["pixel"] for n in names])
            out["results"].append(row)
    if log:
        from .registry import log_run
        allrows = [r for r in out["results"] if r["layout"].startswith("ALL real")]
        log_run("evaluate", "track-sim", model="track", benchmark="Track snap simulation (real page layouts)",
                metrics={f"snap_{r['error_deg']}deg": r["snap"] for r in allrows} |
                        {f"pixel_{r['error_deg']}deg": r["pixel"] for r in allrows},
                details=out, notes=notes or "SIMULATION: snap vs pixel steering, synthetic gaze error")
    return out


def _pooled_ci(arrs, n=2000, seed=1234):
    """95% CI of the equal-weight mean over layouts, bootstrapping layouts (pages), not trials: with 200 trials per
    element the trial-level noise is tiny, and the real uncertainty is which pages were sampled."""
    rng = np.random.default_rng(seed)
    m = np.array([np.asarray(a, float).mean() for a in arrs])
    means = m[rng.integers(0, len(m), (n, len(m)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def print_sim(out, summary_only=False):
    print(f"SIMULATION (not a human measurement). 1 deg = {out['px_per_degree']:.1f} px at {out['distance_cm']:.0f} cm, "
          f"{out['screen_px'][0]} px / {out['screen_cm']} cm screen. bias share {out['bias_share']}, "
          f"fixation {out['fixation']}, assumed voice accuracy {out['voice_accuracy_assumed']}.")
    for name, lay in out["layouts"].items():
        print(f"  layout {name}: {lay['elements']} elements, median {lay['median_w']:.0f}x{lay['median_h']:.0f} px, "
              f"vocabulary {lay['vocabulary']}")
    va = out["voice_accuracy_assumed"]
    print(f"| error | layout | pixel | nearest centre | snap | snap top-3 | top-3 + voice ({va}) | top-3 + perfect voice | "
          f"gaze x voice, all elements ({va}) | voice only ({va}) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in out["results"]:
        if summary_only and not r["layout"].startswith("ALL"):
            continue
        f = lambda k: f"{100 * r[k]:.1f}%" + (f" [{100 * r[k + '_ci'][0]:.1f}, {100 * r[k + '_ci'][1]:.1f}]" if k + "_ci" in r else "")  # noqa: E731
        print(f"| {r['error_deg']}° | {r['layout']} | {f('pixel')} | {f('centre')} | {f('snap')} | {f('topk')} | "
              f"{f('fused')} | {f('fused_perfect_voice')} | {f('fused_soft')} | {f('voice_only')} |")


# ----------------------------------------------------------------------------- MPIIFaceGaze
MPII_DIR = os.path.join(ROOT, "data", "raw", "mpiifacegaze", "MPIIFaceGaze")
MPII_CACHE = os.path.join(ROOT, "data", "cache", "mpiifacegaze_mediapipe.npz")
TASK = os.path.join(ROOT, "app", "static", "vendor", "mediapipe", "face_landmarker.task")
EYES = [(133, 33, 159, 145), (362, 263, 386, 374)]
IRIS = [list(range(468, 473)), list(range(473, 478))]


def load_mpii(root=MPII_DIR):
    """MPIIFaceGaze annotations. Returns a list of dicts: subject, path, screen point, 3D face centre and gaze
    target (camera coordinates, mm), gaze direction (unit vector) and (pitch, yaw) in radians."""
    rows = []
    for s in range(15):
        sub = f"p{s:02d}"
        with open(os.path.join(root, sub, f"{sub}.txt")) as f:
            for line in f:
                p = line.split()
                v = np.array(p[1:27], float)
                fc, gt = v[20:23], v[23:26]
                g = gt - fc
                g = g / np.linalg.norm(g)
                rows.append({"subject": s, "path": os.path.join(root, sub, p[0]), "screen": v[0:2], "face_centre": fc, "target_3d": gt,
                             "gaze": g, "pitch": math.asin(-g[1]), "yaw": math.atan2(-g[0], -g[2]),
                             "head_rt": v[14:20], "eye": p[27]})
    return rows


def _extract_chunk(args):
    paths, task = args
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=task),
                                        running_mode=vision.RunningMode.IMAGE, num_faces=1,
                                        output_facial_transformation_matrixes=True)
    lm_out = np.full((len(paths), 478, 3), np.nan, np.float32)
    mat_out = np.full((len(paths), 4, 4), np.nan, np.float32)
    wh = np.zeros((len(paths), 2), np.int32)
    with vision.FaceLandmarker.create_from_options(opts) as det:
        for i, p in enumerate(paths):
            img = mp.Image.create_from_file(p)
            wh[i] = (img.width, img.height)
            r = det.detect(img)
            if r.face_landmarks:
                lm_out[i] = [(q.x, q.y, q.z) for q in r.face_landmarks[0]]
                mat_out[i] = np.asarray(r.facial_transformation_matrixes[0])
    return lm_out, mat_out, wh


def mpii_extract(workers=6, chunk=250):
    """MediaPipe Face Landmarker on every MPIIFaceGaze image, CPU only. Caches landmarks + transformation matrix."""
    from multiprocessing import Pool
    rows = load_mpii()
    paths = [r["path"] for r in rows]
    chunks = [(paths[i:i + chunk], TASK) for i in range(0, len(paths), chunk)]
    t0 = time.time()
    lms, mats, whs = [], [], []
    with Pool(workers) as pool:
        for i, (lm, mat, wh) in enumerate(pool.imap(_extract_chunk, chunks)):
            lms.append(lm); mats.append(mat); whs.append(wh)
            print(f"{(i + 1) * chunk}/{len(paths)} images, {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.dirname(MPII_CACHE), exist_ok=True)
    np.savez_compressed(MPII_CACHE, lm=np.concatenate(lms), mat=np.concatenate(mats), wh=np.concatenate(whs),
                        paths=np.array([os.path.relpath(p, MPII_DIR) for p in paths]))
    print("saved", MPII_CACHE, f"{time.time() - t0:.0f} s")


def head_pose(M):
    """Same as track.js headPose() for a row-major 4x4 matrix."""
    R, t = M[:3, :3], M[:3, 3]
    fx, fy, fz = R[0, 2], R[1, 2], R[2, 2]
    return [math.atan2(fx, fz), math.asin(max(-1, min(1, -fy))), math.atan2(R[1, 0], R[0, 0]), t[0], t[1], t[2]]


def eye_features(lm, aspect=1.0):
    """Same as track.js features().e: [u, v, open] with x scaled by the image aspect ratio."""
    P = lm[:, :2] * np.array([aspect, 1.0])
    rings = [P[ids].mean(0) for ids in IRIS]
    u = v = op = 0.0
    for inner, outer, top, bottom in EYES:
        a, b = (P[inner], P[outer]) if P[inner][0] <= P[outer][0] else (P[outer], P[inner])
        c = (a + b) / 2
        iris = min(rings, key=lambda q: np.linalg.norm(q - c))
        ev = b - a
        w2 = max(1e-12, ev @ ev)
        d = iris - a
        u += (d @ ev) / w2 / 2
        v += (d[0] * -ev[1] + d[1] * ev[0]) / w2 / 2
        op += np.linalg.norm(P[top] - P[bottom]) / math.sqrt(w2) / 2
    return [u, v, op]


def poly2(e, h, c):
    u, v, o = e[0] - c[0], e[1] - c[1], e[2] - c[2]
    yaw, pitch, roll, tx, ty, tz = h
    z = tz if abs(tz) > 1e-6 else -50
    return [u, v, u * u, v * v, u * v, o, yaw, pitch, roll, tx / z, ty / z, z]


def angular_error_deg(py, yy, pt, yt):
    """Angle between gaze vectors given as (pitch, yaw) in radians, same convention as load_mpii."""
    def vec(p, y):
        return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], -1)
    a, b = vec(py, yy), vec(pt, yt)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def ridge_fit(X, Y, lam, w=None):
    """Weighted ridge on standardised inputs (unpenalised intercept). Returns {mu, sd, B}, B = (1 + d) x outputs."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.ones(len(X)) if w is None else w
    A = (Z * w[:, None]).T @ Z + lam * w.sum() * np.diag([0] + [1] * (Z.shape[1] - 1))
    return {"mu": mu, "sd": sd, "B": np.linalg.solve(A, (Z * w[:, None]).T @ Y)}


def ridge_apply(p, X, clip=None):
    """Same as track.js popPredict(): standardised inputs optionally capped at +-clip."""
    Z = (X - p["mu"]) / p["sd"]
    if clip:
        Z = np.clip(Z, -clip, clip)
    return np.hstack([np.ones((len(Z), 1)), Z]) @ p["B"]


def ridge(X, Y, lam, w=None):
    p = ridge_fit(X, Y, lam, w)
    return lambda Xn: ridge_apply(p, Xn)


POP_EYE_IDS = sorted({133, 33, 159, 145, 362, 263, 386, 374, 160, 158, 144, 153, 385, 387, 373, 380} | set(range(468, 478)))
POP_JSON = os.path.join(ROOT, "app", "static", "track_pop.json")
POP_LAMBDA = 1e-3
POP_CLIP = 5          # standardised population inputs are capped, as in track.js (a glitch frame cannot fling gaze)


def eye_frame(lm, aspect=1.0):
    """Same as track.js eyeFrame(): eye-region landmarks relative to the midpoint of the outer eye corners, scaled by
    their distance and rotated so the corner line is horizontal (removes in-plane roll). 26 points x (x, y)."""
    P = lm[:, :2] * np.array([aspect, 1.0])
    c = (P[33] + P[263]) / 2
    s = np.linalg.norm(P[33] - P[263])
    ang = math.atan2(*(P[263] - P[33])[::-1])
    rot = np.array([[math.cos(-ang), -math.sin(-ang)], [math.sin(-ang), math.cos(-ang)]])
    return (((P[POP_EYE_IDS] - c) / s) @ rot.T).reshape(-1)


def mpii_features():
    rows = load_mpii()
    d = np.load(MPII_CACHE)
    lm, mat, wh = d["lm"], d["mat"], d["wh"]
    ok = ~np.isnan(lm[:, 0, 0])
    E, H = np.zeros((len(rows), 3)), np.zeros((len(rows), 6))
    for i in np.flatnonzero(ok):
        E[i] = eye_features(lm[i], wh[i, 0] / wh[i, 1])
        H[i] = head_pose(mat[i])
    blink = E[:, 2] < 0.12
    subj = np.array([r["subject"] for r in rows])
    target = np.array([[r["pitch"], r["yaw"]] for r in rows])
    screen = np.array([r["screen"] for r in rows])
    # Rich landmark features for the person-independent baseline (eye_frame below).
    R = np.zeros((len(rows), len(POP_EYE_IDS) * 2))
    for i in np.flatnonzero(ok):
        R[i] = eye_frame(lm[i], wh[i, 0] / wh[i, 1])
    return {"ok": ok & ~blink, "E": E, "H": H, "R": R, "subject": subj, "target": target, "screen": screen, "n": len(rows)}


def mpii_eval(lams=(1e-4, 1e-3, 1e-2, 0.1, 1.0), seed=1234, notes="", log=True):
    F = mpii_features()
    ok, subj, tgt = F["ok"], F["subject"], F["target"]
    res = {"images": int(F["n"]), "face_found_open_eyes": int(ok.sum()), "protocol": "leave-one-subject-out, 15 folds",
           "loso": {}, "personal": {}}
    # Person-independent (LOSO): features -> (pitch, yaw). Lambda picked by inner LOSO on the training subjects only.
    feats = {
        "poly2_eye_head (Nullpoint features)": lambda idx, c: np.array([poly2(F["E"][i], F["H"][i], c) for i in idx]),
        "eye_landmarks_head_frame + head pose": lambda idx, c: np.hstack([F["R"][idx], F["H"][idx][:, :3]]),
    }
    mean_pred = {}
    for name, fn in feats.items():
        per_subject = []
        for s in range(15):
            tr = np.flatnonzero(ok & (subj != s))
            te = np.flatnonzero(ok & (subj == s))
            c = np.median(F["E"][tr], 0)
            Xtr, Xte = fn(tr, c), fn(te, c)
            # inner validation: hold out 2 training subjects
            inner = [q for q in range(15) if q != s][:2]
            iv = np.isin(subj[tr], inner)
            best = min(lams, key=lambda lam: angular_error_deg(*ridge(Xtr[~iv], tgt[tr][~iv], lam)(Xtr[iv]).T, *tgt[tr][iv].T).mean())
            pred = ridge(Xtr, tgt[tr], best)(Xte)
            per_subject.append(float(angular_error_deg(pred[:, 0], pred[:, 1], tgt[te, 0], tgt[te, 1]).mean()))
        res["loso"][name] = {"mean_deg": float(np.mean(per_subject)), "per_subject_deg": per_subject}
    # Trivial baseline: predict the training mean gaze direction.
    per_subject = []
    for s in range(15):
        tr, te = ok & (subj != s), ok & (subj == s)
        m = tgt[tr].mean(0)
        per_subject.append(float(angular_error_deg(m[0], m[1], tgt[te, 0], tgt[te, 1]).mean()))
    res["loso"]["mean gaze (no model)"] = {"mean_deg": float(np.mean(per_subject)), "per_subject_deg": per_subject}
    # Personal calibration: for each subject, 9 calibration images whose screen targets are nearest a 3x3 grid
    # (the Nullpoint protocol), fit the Nullpoint poly2 mapping (lambda chosen by leave-one-point-out on those 9),
    # test on all other images of that subject (other days). Also with 9 x 5 images (5 nearest per grid point).
    # Population model (rich landmark features, trained on the other 14 subjects) as features for calibration.
    pop = np.zeros_like(tgt)
    Xr = np.hstack([F["R"], F["H"]])
    for s in range(15):
        tr, te = ok & (subj != s), ok & (subj == s)
        pop[te] = ridge(Xr[tr], tgt[tr], 1e-3)(Xr[te])
    for per_point in (1, 5):
        per_subject, pop_cal, pop_nocal = [], [], []
        for s in range(15):
            idx = np.flatnonzero(ok & (subj == s))
            scr = F["screen"][idx]
            lo, hi = scr.min(0), scr.max(0)
            cal = []
            for fy in (0.1, 0.5, 0.9):
                for fx in (0.1, 0.5, 0.9):
                    g = lo + (hi - lo) * np.array([fx, fy])
                    d = np.linalg.norm(scr - g, axis=1)
                    d[np.isin(np.arange(len(idx)), cal)] = np.inf
                    cal += list(np.argsort(d)[:per_point])
            cal = np.array(cal)
            test = np.setdiff1d(np.arange(len(idx)), cal)
            c = np.median(F["E"][idx[cal]], 0)
            X = np.array([poly2(F["E"][i], F["H"][i], c) for i in idx])
            groups = np.repeat(np.arange(9), per_point)

            def lopo(lam):
                errs = []
                for gq in range(9):
                    trm = groups != gq
                    p = ridge(X[cal[trm]], tgt[idx[cal[trm]]], lam)(X[cal[~trm]])
                    errs.append(angular_error_deg(p[:, 0], p[:, 1], *tgt[idx[cal[~trm]]].T).mean())
                return np.mean(errs)
            best = min(lams, key=lopo)
            p = ridge(X[cal], tgt[idx[cal]], best)(X[test])
            per_subject.append(float(angular_error_deg(p[:, 0], p[:, 1], *tgt[idx[test]].T).mean()))
            # population model + per-person offset (bias correction from the same calibration images)
            off = (tgt[idx[cal]] - pop[idx[cal]]).mean(0)
            q = pop[idx[test]] + off
            pop_cal.append(float(angular_error_deg(q[:, 0], q[:, 1], *tgt[idx[test]].T).mean()))
            q0 = pop[idx[test]]
            pop_nocal.append(float(angular_error_deg(q0[:, 0], q0[:, 1], *tgt[idx[test]].T).mean()))
        res["personal"][f"poly2, 9 points x {per_point} image(s)"] = {"mean_deg": float(np.mean(per_subject)),
                                                                        "per_subject_deg": per_subject}
        res["personal"][f"population model + offset, 9 points x {per_point} image(s)"] = {
            "mean_deg": float(np.mean(pop_cal)), "per_subject_deg": pop_cal}
        res["personal"]["population model, no calibration (same test images)"] = {
            "mean_deg": float(np.mean(pop_nocal)), "per_subject_deg": pop_nocal}
    if log:
        from .registry import log_run
        log_run("evaluate", "track-mpii", model="track", benchmark="MPIIFaceGaze landmark baselines",
                metrics={k: v["mean_deg"] for k, v in {**res["loso"], **res["personal"]}.items()}, details=res,
                notes=notes or "MediaPipe landmark features + ridge; LOSO and 9-point personal calibration")
    return res

def pop_inputs(F):
    """Population head inputs: eye_frame landmarks + the 6 head-pose values."""
    return np.hstack([F["R"], F["H"]])


def pop_screen_x(pp, H):
    """Same as track.js MAPPINGS.pop: population (pitch, yaw) + head translation over depth, mapped linearly to the screen."""
    z = np.where(np.abs(H[:, 5]) > 1e-6, H[:, 5], -50)
    return np.column_stack([pp[:, 0], pp[:, 1], H[:, 3] / z, H[:, 4] / z])


def mpii_screen_eval(lams=(1e-4, 1e-3, 1e-2, 0.1, 1.0), notes="", log=True):
    """The shipped pipeline on MPIIFaceGaze: per subject, 9 calibration images nearest a 3 x 3 grid (x 1 or x 5),
    each mapping fitted to SCREEN PIXELS with lambda chosen by leave-one-point-out, inputs capped at +-3 SD as in
    track.js, tested on all the subject's other images. The population head is trained leave-one-subject-out. Pixel
    errors are converted to mm with the subject's screen size and to degrees with the face-to-target distance."""
    from scipy.io import loadmat
    rows = load_mpii()
    F = mpii_features()
    ok, subj = F["ok"], F["subject"]
    X = pop_inputs(F)
    pop = np.zeros((F["n"], 2))
    for s in range(15):
        tr, te = ok & (subj != s), ok & (subj == s)
        pop[te] = ridge_apply(ridge_fit(X[tr], F["target"][tr], POP_LAMBDA), X[te], POP_CLIP)
    res = {"protocol": "per subject: 9-point calibration to screen pixels, LOPO lambda, inputs capped at 3 SD; "
                       "population head leave-one-subject-out", "rows": {}}
    raw = {}
    for s in range(15):
        m = loadmat(os.path.join(MPII_DIR, f"p{s:02d}", "Calibration", "screenSize.mat"))
        raw[s] = float(np.squeeze(m["width_mm"])) / float(np.squeeze(m["width_pixel"]))
    for per_point in (1, 5):
        out = {"poly2": [], "pop": [], "selected by LOPO (as Nullpoint)": []}
        picked_pop = 0
        for s in range(15):
            idx = np.flatnonzero(ok & (subj == s))
            scr = F["screen"][idx]
            lo, hi = scr.min(0), scr.max(0)
            cal = []
            for fy in (0.1, 0.5, 0.9):
                for fx in (0.1, 0.5, 0.9):
                    g = lo + (hi - lo) * np.array([fx, fy])
                    d = np.linalg.norm(scr - g, axis=1)
                    d[np.isin(np.arange(len(idx)), cal)] = np.inf
                    cal += list(np.argsort(d)[:per_point])
            cal = np.array(cal)
            test = np.setdiff1d(np.arange(len(idx)), cal)
            groups = np.repeat(np.arange(9), per_point)
            dmm = np.array([np.linalg.norm(rows[i]["target_3d"] - rows[i]["face_centre"]) for i in idx])
            c = np.median(F["E"][idx[cal]], 0)
            designs = {"poly2": np.array([poly2(F["E"][i], F["H"][i], c) for i in idx]),
                       "pop": pop_screen_x(pop[idx], F["H"][idx])}
            errs, lopo_err = {}, {}
            for name, D in designs.items():
                def lopo(lam):
                    e = []
                    for gq in range(9):
                        trm = groups != gq
                        q = ridge_apply(ridge_fit(D[cal[trm]], scr[cal[trm]], lam), D[cal[~trm]], 3)
                        e.append(np.linalg.norm(np.median(q, 0) - scr[cal[~trm]].mean(0)))
                    return float(np.mean(e))
                scores = {lam: lopo(lam) for lam in lams}
                best = min(scores, key=scores.get)
                lopo_err[name] = scores[best]
                q = ridge_apply(ridge_fit(D[cal], scr[cal], best), D[test], 3)
                emm = np.linalg.norm(q - scr[test], axis=1) * raw[s]
                errs[name] = float(np.degrees(np.arctan(emm / dmm[test])).mean())
                out[name].append(errs[name])
            pick = min(lopo_err, key=lopo_err.get)
            picked_pop += pick == "pop"
            out["selected by LOPO (as Nullpoint)"].append(errs[pick])
        for name, v in out.items():
            res["rows"][f"{name}, 9 points x {per_point} image(s)"] = {"mean_deg": float(np.mean(v)), "per_subject_deg": v}
        res["rows"][f"selected by LOPO (as Nullpoint), 9 points x {per_point} image(s)"]["pop_chosen"] = f"{picked_pop}/15"
    if log:
        from .registry import log_run
        log_run("evaluate", "track-mpii-screen", model="track", benchmark="MPIIFaceGaze screen-space calibration",
                metrics={k: v["mean_deg"] for k, v in res["rows"].items()}, details=res,
                notes=notes or "Shipped pipeline (features -> screen px, 9-point calibration) on MPIIFaceGaze: poly2 vs "
                               "population head (LOSO) + per-person linear map; degrees from screen mm and face distance")
    return res


def export_pop(path=POP_JSON):
    """Population head on all 15 MPIIFaceGaze subjects -> JSON for track.js. CC BY-NC-SA 4.0 data: non-commercial."""
    F = mpii_features()
    ok = F["ok"]
    p = ridge_fit(pop_inputs(F)[ok], F["target"][ok], POP_LAMBDA)
    out = {"version": 1, "eye_ids": POP_EYE_IDS, "clip": POP_CLIP, "lambda": POP_LAMBDA,
           "inputs": "eye_frame(landmarks) (26 x 2) + head pose [yaw, pitch, roll, tx, ty, tz]", "outputs": ["pitch", "yaw"],
           "mu": p["mu"].tolist(), "sd": p["sd"].tolist(), "B": p["B"].tolist(), "n_images": int(ok.sum()),
           "trained_on": "MPIIFaceGaze (Zhang et al., CVPRW 2017), 15 subjects, MediaPipe Face Landmarker features",
           "licence": "Derived from MPIIFaceGaze, CC BY-NC-SA 4.0: non-commercial use, share alike, attribute."}
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", path, f"{os.path.getsize(path) / 1024:.1f} KB, {out['n_images']} images")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sim", "mpii-extract", "mpii", "mpii-screen", "export-pop"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--bias-share", type=float, default=0.0)
    ap.add_argument("--fixation", default="centre50", choices=["centre50", "uniform"])
    ap.add_argument("--voice-acc", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--variants", action="store_true", help="also run the bias and fixation sensitivity variants")
    ap.add_argument("--json", help="write the full result to this file")
    a = ap.parse_args()
    if a.cmd == "sim":
        out = run_sim(reps=a.reps, bias_share=a.bias_share, fixation=a.fixation, voice_acc=a.voice_acc, log=not a.no_log)
        print_sim(out)
        if a.variants:  # sensitivity to the error and fixation model (not logged)
            out = {"main": out, "variants": {}}
            for bias, fix in [(0.5, "centre50"), (0.0, "uniform")]:
                v = run_sim(reps=a.reps, bias_share=bias, fixation=fix, voice_acc=a.voice_acc, log=False)
                out["variants"][f"bias{bias}_{fix}"] = v
                print()
                print_sim(v, summary_only=True)
    elif a.cmd == "mpii-extract":
        mpii_extract(workers=a.workers)
        return
    elif a.cmd == "export-pop":
        export_pop()
        return
    elif a.cmd == "mpii-screen":
        out = mpii_screen_eval(log=not a.no_log)
        for k, v in out["rows"].items():
            print(f"{k}: {v['mean_deg']:.2f} deg" + (f" (pop chosen {v['pop_chosen']})" if "pop_chosen" in v else ""))
    else:
        out = mpii_eval(log=not a.no_log)
        print(json.dumps({k: v for k, v in out.items()}, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
