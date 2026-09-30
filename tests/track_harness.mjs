// Runs the pure parts of app/static/track.js in Node for tests/test_track.py.
// Usage: node tests/track_harness.mjs < input.json   (prints a JSON result)
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const T = await import(pathToFileURL(path.join(here, "..", "app", "static", "track.js")).href);
const inp = JSON.parse(fs.readFileSync(0, "utf8"));
const out = {};

// Seeded RNG so results are reproducible.
let seed = 12345;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

// 1. Posterior on given points/rects (compared with the Python implementation).
if (inp.posterior) {
  const { points, rects, sigma } = inp.posterior;
  const targets = rects.map((r, i) => ({ id: i, label: "e" + i, rect: { left: r[0], top: r[1], width: r[2], height: r[3] } }));
  out.posterior = points.map(([x, y]) => {
    const p = T.posterior(x, y, targets, sigma, { width: 1920, height: 1080 }, 0.05);
    const arr = new Array(rects.length).fill(0);
    p.items.forEach((it) => (arr[it.id] = it.p));
    return { p: arr, p_none: p.p_none };
  });
}

// 2. Eye features on given landmark sets (compared with Python eye_features).
if (inp.features) {
  out.features = inp.features.map(({ lm, aspect }) => {
    const f = T.features(lm.map(([x, y, z]) => ({ x, y, z })), null, aspect);
    return f ? f.e : null;
  });
}

// 3. Calibration on synthetic data: a known nonlinear eye -> screen map with head motion and noise.
if (inp.calibration) {
  const truth = (u, v, yaw, pitch) => [960 + 3000 * u + 1800 * u * u + 900 * yaw, 540 + 2600 * v - 700 * v * v + 800 * pitch];
  const inv = (sx, sy, yaw, pitch) => {           // numerically invert truth for the eye features
    let u = 0, v = 0;
    for (let i = 0; i < 50; i++) {
      const [x, y] = truth(u, v, yaw, pitch);
      u -= (x - sx) / (3000 + 3600 * u); v -= (y - sy) / (2600 - 1400 * v);
    }
    return [u, v];
  };
  const frame = (sx, sy, noise) => {
    const yaw = 0.05 * gauss(), pitch = 0.05 * gauss();
    const [u, v] = inv(sx, sy, yaw, pitch);
    return { e: [0.5 + u + noise * gauss(), v + noise * gauss(), 0.3], h: [yaw, pitch, 0, 0, 0, -55], v0: [u, v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], t: 0 };
  };
  const pts = [];
  for (const fy of [0.1, 0.5, 0.9]) for (const fx of [0.1, 0.5, 0.9]) {
    const frames = []; for (let i = 0; i < 25; i++) frames.push(frame(fx * 1920, fy * 1080, inp.calibration.noise));
    frames.push({ ...frames[0], e: [5, 5, 0.3] });                 // one glitch frame: must be rejected
    pts.push({ x: fx * 1920, y: fy * 1080, frames });
  }
  const res = T.calibrate(pts);
  const errs = [];
  for (const [fx, fy] of [[0.3, 0.3], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.7, 0.7]]) {
    const fr = []; for (let i = 0; i < 30; i++) fr.push(frame(fx * 1920, fy * 1080, inp.calibration.noise));
    const pr = fr.map((f) => T.predict(res.models.poly2, f));
    errs.push(Math.hypot(T.median(pr.map((q) => q[0])) - fx * 1920, T.median(pr.map((q) => q[1])) - fy * 1080));
  }
  out.calibration = { kind: res.kind, lopo_px: res.lopo_px, frames: res.frames, heldout_px: errs.reduce((a, b) => a + b, 0) / errs.length };
}

// 4. Hysteresis: gaze jitters around the border between two adjacent buttons.
if (inp.hysteresis) {
  const targets = [{ id: "a", label: "save", rect: { left: 900, top: 500, width: 60, height: 30 } },
                   { id: "b", label: "share", rect: { left: 968, top: 500, width: 60, height: 30 } }];
  const run = (opts) => {
    const s = new T.Snapper({ sigma: 40, ...opts }); s.setTargets(targets);
    const f = new T.OneEuro();
    let switches = 0, prev = null;
    for (let i = 0; i < 300; i++) {                                // 10 s at 30 fps, looking at x = 962 (the gap)
      const t = i * 33.3;
      const [x, y] = f.filter([962 + 25 * gauss(), 515 + 25 * gauss()], t);
      const st = s.update(x, y, t, true);
      if (st.highlight !== prev && prev !== null) switches++;
      prev = st.highlight;
    }
    return switches;
  };
  out.hysteresis = { with: run({}), without: run({ switchRatio: 1.0, switchMs: 0, dropMs: 0 }) };
}

// 5. Dwell + candidates JSON + fusion.
if (inp.dwell) {
  const s = new T.Snapper({ sigma: 30, dwellMs: 500 });
  s.setTargets([{ n: 1, label: "Sign in", rect: { left: 100, top: 100, width: 80, height: 30 } },
                { n: 2, label: "Register", rect: { left: 190, top: 100, width: 80, height: 30 } },
                { n: 3, label: "Help", rect: { left: 280, top: 100, width: 80, height: 30 } }]);
  const fx = new T.Fixation(100, 150);
  let dwellAt = null;
  for (let i = 0; i < 40; i++) {
    const t = i * 33.3, st = s.update(182 + gauss() * 2, 115 + gauss() * 2, t, fx.push(182, 115, t).fixating);  // near the border
    if (st.dwell && dwellAt === null) dwellAt = t;
  }
  const c = s.candidates(3);
  out.dwell = { dwellAt, candidates: c, fused: T.fuseVoice(c, { register: 0.9, "sign in": 0.05 }) };
}

// 6. One Euro reduces jitter on a still target and follows a step.
if (inp.oneeuro) {
  const f = new T.OneEuro();
  let raw = 0, sm = 0, prevR = null, prevS = null, after = null;
  for (let i = 0; i < 200; i++) {
    const t = i * 33.3, tx = i < 100 ? 500 : 1400;
    const r = [tx + 30 * gauss(), 500 + 30 * gauss()], s = f.filter(r, t);
    if (prevR && i < 100) { raw += Math.hypot(r[0] - prevR[0], r[1] - prevR[1]); sm += Math.hypot(s[0] - prevS[0], s[1] - prevS[1]); }
    if (i === 115) after = s[0];
    prevR = r; prevS = s;
  }
  out.oneeuro = { raw_step: raw / 99, smooth_step: sm / 99, x_500ms_after_step: after };
}

process.stdout.write(JSON.stringify(out));
