/* Afono Track Model v0: webcam gaze.
   MediaPipe face landmarks -> eye/iris/head features -> ridge regression fitted by a
   9-point calibration -> One Euro smoothing. Scored with accuracy and precision
   (pixels and degrees) and a snap-to-target selection test. */
import { FaceLandmarker, FilesetResolver } from "/vendor/mediapipe/vision_bundle.mjs";

const EYES = [
  { inner: 133, outer: 33, top: 159, bottom: 145 },
  { inner: 362, outer: 263, top: 386, bottom: 374 },
];
const IRIS = [468, 473];

let landmarker = null, video = null, running = false, last = null, model = null, onFrame = null;
let fps = 0, frames = 0, fpsT = performance.now();

function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
function len(a) { return Math.hypot(a.x, a.y); }

/** Gaze features from one frame's landmarks, or null during a blink / no face. */
export function features(lm, matrix) {
  const irisPts = IRIS.map((i) => lm[i]);
  const eyes = EYES.map((e) => ({ inner: lm[e.inner], outer: lm[e.outer], top: lm[e.top], bottom: lm[e.bottom] }));
  const f = [];
  let open = 0, ubar = 0, vbar = 0;
  eyes.forEach((e) => {
    const c = { x: (e.inner.x + e.outer.x) / 2, y: (e.inner.y + e.outer.y) / 2 };
    const iris = irisPts.reduce((best, p) => (len(sub(p, c)) < len(sub(best, c)) ? p : best));
    const ev = sub(e.outer, e.inner), w = len(ev);
    const d = sub(iris, e.inner);
    const u = (d.x * ev.x + d.y * ev.y) / (w * w);
    const v = (iris.y - (e.top.y + e.bottom.y) / 2) / w;
    open += Math.abs(e.top.y - e.bottom.y) / w;
    f.push(u, v); ubar += u / 2; vbar += v / 2;
  });
  if (open / 2 < 0.12) return null;                                   // blink
  const nose = lm[1], scale = len(sub(lm[33], lm[263]));
  const m = matrix ? matrix.data : null;
  f.push(nose.x, nose.y, scale, m ? m[8] : 0, m ? m[9] : 0, ubar * ubar, vbar * vbar, ubar * vbar);
  return f;
}

/* ---------- ridge regression on standardised features ---------- */
function solve(A, b) {
  const n = b.length;
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
    for (let r = i + 1; r < n; r++) {
      const k = A[r][i] / A[i][i];
      for (let c = i; c < n; c++) A[r][c] -= k * A[i][c];
      b[r] -= k * b[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let c = i + 1; c < n; c++) s -= A[i][c] * x[c]; x[i] = s / A[i][i]; }
  return x;
}
export function fit(X, Y, lambda = 1.0) {
  const d = X[0].length, mu = new Array(d).fill(0), sd = new Array(d).fill(0);
  X.forEach((r) => r.forEach((v, j) => (mu[j] += v / X.length)));
  X.forEach((r) => r.forEach((v, j) => (sd[j] += (v - mu[j]) ** 2 / X.length)));
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  const Z = X.map((r) => [1, ...r.map((v, j) => (v - mu[j]) / sd[j])]);
  const D = d + 1, W = [];
  for (let k = 0; k < 2; k++) {
    const A = Array.from({ length: D }, () => new Array(D).fill(0)), b = new Array(D).fill(0);
    Z.forEach((z, i) => { for (let r = 0; r < D; r++) { b[r] += z[r] * Y[i][k]; for (let c = 0; c < D; c++) A[r][c] += z[r] * z[c]; } });
    for (let r = 1; r < D; r++) A[r][r] += lambda;
    W.push(solve(A, b));
  }
  return { mu, sd, W, n: X.length };
}
export function predict(m, f) {
  const z = [1, ...f.map((v, j) => (v - m.mu[j]) / m.sd[j])];
  return m.W.map((w) => w.reduce((s, wi, i) => s + wi * z[i], 0));
}

/* ---------- One Euro filter (Casiez et al., 2012) ---------- */
export class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.015, dCutoff = 1) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0, t: null }); }
  alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(v, t) {
    if (this.x === null) { this.x = v; this.t = t; return v; }
    const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
    const dv = v.map((x, i) => (x - this.x[i]) / dt);
    const ad = this.alpha(this.dCutoff, dt);
    this.dx = dv.map((d, i) => ad * d + (1 - ad) * (Array.isArray(this.dx) ? this.dx[i] : 0));
    this.x = v.map((x, i) => { const a = this.alpha(this.minCutoff + this.beta * Math.abs(this.dx[i]), dt); return a * x + (1 - a) * this.x[i]; });
    return this.x;
  }
}

/* ---------- camera + landmarker ---------- */
export async function start(videoEl, status) {
  video = videoEl;
  status("Loading the Track Model…");
  const files = await FilesetResolver.forVisionTasks("/vendor/mediapipe/wasm");
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: "/vendor/mediapipe/face_landmarker.task", delegate },
    runningMode: "VIDEO", numFaces: 1, outputFacialTransformationMatrixes: true });
  try { landmarker = await FaceLandmarker.createFromOptions(files, opts("GPU")); }
  catch (e) { landmarker = await FaceLandmarker.createFromOptions(files, opts("CPU")); }
  status("Starting the camera…");
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
  video.srcObject = stream; await video.play();
  running = true; loop();
}
export function stop() {
  running = false;
  if (video && video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
}
function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    const r = landmarker.detectForVideo(video, performance.now());
    const lm = r.faceLandmarks && r.faceLandmarks[0];
    last = lm ? { f: features(lm, r.facialTransformationMatrixes && r.facialTransformationMatrixes[0]), t: performance.now(), face: true }
              : { f: null, t: performance.now(), face: false };
    frames++;
    if (performance.now() - fpsT > 1000) { fps = frames; frames = 0; fpsT = performance.now(); }
    if (onFrame) onFrame(last);
  }
  requestAnimationFrame(loop);
}
export function info() { return { fps, face: !!(last && last.face), open: !!(last && last.f), calibrated: !!model }; }
export function setModel(m) { model = m; }
export function getModel() { return model; }
export function subscribe(fn) { onFrame = fn; }

/** Collect features for `ms` milliseconds after `settle` ms. */
export function collect(ms, settle = 450) {
  return new Promise((resolve) => {
    const out = [], t0 = performance.now();
    const prev = onFrame;
    onFrame = (fr) => {
      const t = performance.now() - t0;
      if (t > settle && fr.f) out.push(fr.f);
      if (t > settle + ms) { onFrame = prev; resolve(out); }
      if (prev) prev(fr);
    };
  });
}

/** Accuracy (distance of the median estimate to the target) and precision (RMS sample-to-sample). */
export function score(points, pxPerCm, distCm) {
  const deg = (px) => (Math.atan(px / pxPerCm / distCm) * 180) / Math.PI;
  const acc = [], prec = [];
  points.forEach((p) => {
    if (!p.preds.length) return;
    const xs = p.preds.map((q) => q[0]).sort((a, b) => a - b), ys = p.preds.map((q) => q[1]).sort((a, b) => a - b);
    const mx = xs[xs.length >> 1], my = ys[ys.length >> 1];
    acc.push(Math.hypot(mx - p.x, my - p.y));
    let s = 0; for (let i = 1; i < p.preds.length; i++) s += (p.preds[i][0] - p.preds[i - 1][0]) ** 2 + (p.preds[i][1] - p.preds[i - 1][1]) ** 2;
    prec.push(Math.sqrt(s / Math.max(1, p.preds.length - 1)));
  });
  const mean = (a) => a.reduce((s, v) => s + v, 0) / Math.max(1, a.length);
  return { points: acc.length, accuracy_px: mean(acc), precision_px: mean(prec), accuracy_deg: deg(mean(acc)), precision_deg: deg(mean(prec)) };
}
