"""Afono Track Model: tests for app/static/track.js (run in Node) and its Python mirror in semg/gaze_eval.py."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from semg import gaze_eval as G

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HARNESS = os.path.join(ROOT, "tests", "track_harness.mjs")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def js(payload):
    out = subprocess.run(["node", HARNESS], input=json.dumps(payload), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_posterior_matches_python():
    rng = np.random.default_rng(0)
    rects = np.column_stack([rng.uniform(0, 1700, 30), rng.uniform(0, 1000, 30), rng.uniform(20, 200, 30), rng.uniform(15, 60, 30)])
    pts = np.column_stack([rng.uniform(0, 1920, 50), rng.uniform(0, 1080, 50)])
    r = js({"posterior": {"points": pts.tolist(), "rects": rects.tolist(), "sigma": 70.0}})["posterior"]
    post, none = G.posterior(pts[:, 0], pts[:, 1], rects, 70.0)
    np.testing.assert_allclose(np.array([q["p"] for q in r]), post, atol=1e-5)
    np.testing.assert_allclose(np.array([q["p_none"] for q in r]), none, atol=1e-5)


def _face(du=0.0, dv=0.0, rng=None):
    """478 synthetic landmarks: two eyes with corners, lids and irises; iris shifted by du, dv (in eye widths)."""
    lm = np.tile([0.5, 0.5, 0.0], (478, 1)) + (rng.normal(0, 1e-3, (478, 3)) if rng is not None else 0)
    for inner, outer, top, bottom, iris0, cx in [(133, 33, 159, 145, 468, 0.40), (362, 263, 386, 374, 473, 0.60)]:
        w = 0.08
        left, right = cx - w / 2, cx + w / 2
        # image-left eye (subject's right): outer corner on the left; image-right eye: inner corner on the left
        lm[outer if cx < 0.5 else inner, :2] = [left, 0.45]
        lm[inner if cx < 0.5 else outer, :2] = [right, 0.45]
        lm[top, :2] = [cx, 0.43]
        lm[bottom, :2] = [cx, 0.47]
        c = np.array([cx + du * w, 0.45 + dv * w])
        lm[iris0, :2] = c
        for k, off in enumerate([(0.01, 0), (0, -0.01), (-0.01, 0), (0, 0.01)]):
            lm[iris0 + 1 + k, :2] = c + off
    return lm


def test_features_match_python_and_move_together():
    rng = np.random.default_rng(1)
    faces = [_face(0, 0), _face(0.1, 0), _face(0, 0.05), _face(0.05, -0.03, rng)]
    r = js({"features": [{"lm": f.tolist(), "aspect": 4 / 3} for f in faces]})["features"]
    for f, e in zip(faces, r):
        np.testing.assert_allclose(e, G.eye_features(f, 4 / 3), atol=1e-9)
    # Looking right moves both irises right: u must increase (it used to cancel between the two eyes).
    assert r[1][0] - r[0][0] > 0.05
    assert r[2][1] - r[0][1] > 0.02


def test_population_head_matches_python_and_is_offered_by_calibration():
    with open(G.POP_JSON) as f:
        model = json.load(f)
    rng = np.random.default_rng(2)
    faces = []
    for du, dv, roll in [(0, 0, 0.0), (0.1, 0.02, 0.1), (-0.05, -0.03, -0.2)]:
        lm = _face(du, dv, rng)
        c = np.array([0.5, 0.45])                                   # roll the face in the image plane
        R2 = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
        lm[:, :2] = (lm[:, :2] - c) @ R2.T + c
        yaw, pitch = rng.normal(0, 0.1, 2)
        M = np.eye(4)
        M[:3, :3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
        M[:3, :3] = M[:3, :3] @ [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
        M[:3, 3] = [rng.normal(0, 2), rng.normal(0, 2), -55 + rng.normal(0, 3)]
        faces.append((lm, M))
    r = js({"pop": {"model": model, "faces": [{"lm": lm.tolist(), "matrix": M.reshape(-1).tolist(), "aspect": 4 / 3}
                                              for lm, M in faces]}})
    p = {k: np.array(model[k]) for k in ("mu", "sd", "B")}
    for (lm, M), got in zip(faces, r["pop"]):
        np.testing.assert_allclose(got["h"], G.head_pose(M), atol=1e-9)
        x = np.concatenate([G.eye_frame(lm, 4 / 3), G.head_pose(M)])[None]
        np.testing.assert_allclose(got["pop"], G.ridge_apply(p, x, model["clip"])[0], atol=1e-9)
    assert r["pop_calibration"]["kinds"] == ["poly2", "v0", "pop"]
    assert r["pop_calibration"]["kind"] == "pop"                     # the screen is linear in the angles here


def test_calibration_fits_synthetic_mapping_and_rejects_glitch():
    r = js({"calibration": {"noise": 0.002}})["calibration"]
    assert r["frames"] <= 9 * 25            # the glitch frame at every point was rejected
    assert r["heldout_px"] < 40, r          # a 2nd-order map + head pose recovers the synthetic truth
    assert r["lopo_px"]["poly2"] < r["lopo_px"]["v0"]


def test_hysteresis_reduces_flicker():
    r = js({"hysteresis": True})["hysteresis"]
    assert r["with"] < r["without"] / 3, r


def test_dwell_candidates_and_voice_fusion():
    r = js({"dwell": True})["dwell"]
    assert r["dwellAt"] is not None and 450 <= r["dwellAt"] <= 800
    c = r["candidates"]
    assert c["highlight"] == 1 and c["candidates"][0]["label"] == "Sign in"
    assert abs(sum(x["p"] for x in c["candidates"]) + c["p_none"] - 1) < 1e-3
    assert r["fused"][0]["label"] == "Register"   # voice picks among the gaze candidates


def test_one_euro_smooths_and_follows():
    r = js({"oneeuro": True})["oneeuro"]
    assert r["smooth_step"] < r["raw_step"] / 2
    assert abs(r["x_500ms_after_step"] - 1400) < 100


def test_simulation_snap_beats_pixel():
    items = G.synthetic_layouts()["toolbar_32px"]
    rng = np.random.default_rng(0)
    r = G.simulate_layout(items, 2.0, 50, rng)
    assert r["snap"].mean() > r["pixel"].mean()
    assert r["topk"].mean() >= r["snap"].mean()


def test_angular_error_convention():
    assert G.angular_error_deg(0.0, 0.0, 0.0, np.radians(10)) == pytest.approx(10, abs=1e-6)
    assert G.angular_error_deg(np.radians(5), 0.0, 0.0, 0.0) == pytest.approx(5, abs=1e-6)
