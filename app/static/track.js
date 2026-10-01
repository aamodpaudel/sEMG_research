/* Afono Track Model v1: webcam gaze that snaps to page elements.
   MediaPipe Face Landmarker (iris landmarks 468-477 + facial transformation matrix)
   -> eye-in-head features (iris centre relative to the eye corners) + head pose
   -> ridge-regularised 2nd-order polynomial mapping fitted by a 9-point calibration
   -> One Euro smoothing -> fixation detector -> Bayesian snap to the nearest page element
   with hysteresis, and top-k candidates with probabilities for fusion with the voice decoder.

   The pure functions (features, calibrate, predict, OneEuro, Fixation, Snapper, fuseVoice) have no
   browser dependencies so tests/test_track.py can run them in Node. MediaPipe is imported lazily in start(). */

const EYES = [
  { inner: 133, outer: 33, top: 159, bottom: 145 },
  { inner: 362, outer: 263, top: 386, bottom: 374 },
];
const IRIS = [[468, 469, 470, 471, 472], [473, 474, 475, 476, 477]];   // centre + 4 ring points per eye

let landmarker = null, video = null, running = false, last = null, model = null, onFrame = null;
let fps = 0, frames = 0, fpsT = 0;
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
function len(a) { return Math.hypot(a.x, a.y); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / Math.max(1, a.length); }
export function median(a) {
  if (!a.length) return NaN;
  const s = [...a].sort((p, q) => p - q), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/* ---------- features ---------- */

/** Head pose from MediaPipe's 4x4 facial transformation matrix. Handles row- or column-major data. */
export function headPose(matrix) {
  if (!matrix || !matrix.data || matrix.data.length < 16) return [0, 0, 0, 0, 0, 0];
  const m = matrix.data;
  const colMajor = Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11]) < 1e-6;   // translation lives in m[12..14]
  const R = (r, c) => (colMajor ? m[c * 4 + r] : m[r * 4 + c]);
  const t = colMajor ? [m[12], m[13], m[14]] : [m[3], m[7], m[11]];
  const fx = R(0, 2), fy = R(1, 2), fz = R(2, 2);                  // face forward axis in camera space
  const yaw = Math.atan2(fx, fz), pitch = Math.asin(Math.max(-1, Math.min(1, -fy))), roll = Math.atan2(R(1, 0), R(0, 0));
  return [yaw, pitch, roll, t[0], t[1], t[2]];
}

/** The v0 feature vector (kept so every session can compare the old mapping with the new one). */
export function featuresV0(lm, matrix) {
  const irisPts = [468, 473].map((i) => lm[i]);
  const eyes = EYES.map((e) => ({ inner: lm[e.inner], outer: lm[e.outer], top: lm[e.top], bottom: lm[e.bottom] }));
  const f = [];
  let ubar = 0, vbar = 0;
  eyes.forEach((e) => {
    const c = { x: (e.inner.x + e.outer.x) / 2, y: (e.inner.y + e.outer.y) / 2 };
    const iris = irisPts.reduce((best, p) => (len(sub(p, c)) < len(sub(best, c)) ? p : best));
    const ev = sub(e.outer, e.inner), w = len(ev), d = sub(iris, e.inner);
    const u = (d.x * ev.x + d.y * ev.y) / (w * w), v = (iris.y - (e.top.y + e.bottom.y) / 2) / w;
    f.push(u, v); ubar += u / 2; vbar += v / 2;
  });
  const nose = lm[1], scale = len(sub(lm[33], lm[263]));
  const m = matrix && matrix.data ? matrix.data : null;
  f.push(nose.x, nose.y, scale, m ? m[8] : 0, m ? m[9] : 0, ubar * ubar, vbar * vbar, ubar * vbar);
  return f;
}

/** Gaze features from one frame's landmarks, or null during a blink / no face.
    e = [u, v, open]: iris centre (mean of the 5 iris points) along the eye-corner axis (u, 0 = image-left corner,
    1 = image-right corner) and perpendicular to it (v), both in eye widths, averaged over the two eyes; open = lid gap.
    aspect = video width / height, so distances are in square pixels. Being relative to the corners, e is invariant to head translation, scale and in-plane roll.
    h = [yaw, pitch, roll, tx, ty, tz] from the facial transformation matrix (head-pose compensation). */
export function features(lm, matrix, aspect = 1, blink = 0.12) {
  if (!lm || lm.length < 478) return null;
  const raw = lm;
  if (aspect !== 1) lm = lm.map((p) => ({ x: p.x * aspect, y: p.y }));        // normalised coords -> square pixels
  let u = 0, v = 0, open = 0;
  for (const e of EYES) {
    // Orient every eye's axis left -> right in the image, so conjugate eye movements move u the same way in both
    // eyes (measured from the inner corner, the two eyes' u would cancel when averaged).
    const [inner, outer] = lm[e.inner].x <= lm[e.outer].x ? [lm[e.inner], lm[e.outer]] : [lm[e.outer], lm[e.inner]];
    const c = { x: (inner.x + outer.x) / 2, y: (inner.y + outer.y) / 2 };
    // Pick the iris ring whose centre is nearest this eye (robust to MediaPipe's left/right naming).
    const rings = IRIS.map((ids) => ({ x: mean(ids.map((i) => lm[i].x)), y: mean(ids.map((i) => lm[i].y)) }));
    const iris = rings.reduce((best, p) => (len(sub(p, c)) < len(sub(best, c)) ? p : best));
    const ev = sub(outer, inner), w = len(ev) || 1e-6, d = sub(iris, inner);
    u += (d.x * ev.x + d.y * ev.y) / (w * w) / 2;
    v += (d.x * -ev.y + d.y * ev.x) / (w * w) / 2;                          // perpendicular, + = down
    open += len(sub(lm[e.top], lm[e.bottom])) / w / 2;
  }
  if (open < blink) return null;                                            // blink
  return { e: [u, v, open], h: headPose(matrix), v0: featuresV0(raw, matrix) };
}

/* ---------- design matrices ---------- */
export const MAPPINGS = {
  /** v0: the original features, linear ridge. */
  v0: (f) => f.v0,
  /** poly2: 2nd-order polynomial in centred eye features + linear head pose (translation divided by depth). */
  poly2: (f, c) => {
    const u = f.e[0] - c[0], v = f.e[1] - c[1], o = f.e[2] - c[2];
    const [yaw, pitch, roll, tx, ty, tz] = f.h, z = Math.abs(tz) > 1e-6 ? tz : -50;
    return [u, v, u * u, v * v, u * v, o, yaw, pitch, roll, tx / z, ty / z, z];
  },
};

/* ---------- weighted ridge regression on standardised features ---------- */
function solve(A, b) {
  const n = b.length;
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
    const piv = A[i][i] || 1e-12;
    for (let r = i + 1; r < n; r++) {
      const k = A[r][i] / piv;
      for (let c = i; c < n; c++) A[r][c] -= k * A[i][c];
      b[r] -= k * b[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let c = i + 1; c < n; c++) s -= A[i][c] * x[c]; x[i] = s / (A[i][i] || 1e-12); }
  return x;
}

/** Ridge fit of Y (n x 2) on X (n x d). w: per-row weights (default 1). lambda is relative to the total weight. */
export function fit(X, Y, lambda = 1.0, w = null) {
  const n = X.length, d = X[0].length;
  w = w || new Array(n).fill(1);
  const W = w.reduce((s, v) => s + v, 0);
  const mu = new Array(d).fill(0), sd = new Array(d).fill(0);
  X.forEach((r, i) => r.forEach((v, j) => (mu[j] += (w[i] * v) / W)));
  X.forEach((r, i) => r.forEach((v, j) => (sd[j] += (w[i] * (v - mu[j]) ** 2) / W)));
  for (let j = 0; j < d; j++) sd[j] = sd[j] > 1e-12 ? Math.sqrt(sd[j]) : 1;
  const Z = X.map((r) => [1, ...r.map((v, j) => (v - mu[j]) / sd[j])]);
  const D = d + 1, coef = [];
  for (let k = 0; k < 2; k++) {
    const A = Array.from({ length: D }, () => new Array(D).fill(0)), b = new Array(D).fill(0);
    Z.forEach((z, i) => { for (let r = 0; r < D; r++) { b[r] += w[i] * z[r] * Y[i][k]; for (let c = 0; c < D; c++) A[r][c] += w[i] * z[r] * z[c]; } });
    for (let r = 1; r < D; r++) A[r][r] += lambda * W;
    coef.push(solve(A, b));
  }
  return { mu, sd, W: coef, n };
}
export function predictRaw(m, x) {
  const z = [1, ...x.map((v, j) => (v - m.mu[j]) / m.sd[j])];
  return m.W.map((w) => w.reduce((s, wi, i) => s + wi * z[i], 0));
}

/* ---------- 9-point calibration ---------- */
export const LAMBDAS = [1e-4, 1e-3, 1e-2, 0.1, 1];

/** Keep the frames of one calibration point whose eye features are within `z` robust SDs of the point's median
    (drops saccades, half-blinks and tracking glitches). */
export function inliers(frames, z = 3.5) {
  if (frames.length < 5) return frames;
  const keep = new Array(frames.length).fill(true);
  for (let j = 0; j < 2; j++) {
    const xs = frames.map((f) => f.e[j]), m = median(xs);
    const mad = median(xs.map((x) => Math.abs(x - m))) * 1.4826 || 1e-9;
    xs.forEach((x, i) => { if (Math.abs(x - m) / mad > z) keep[i] = false; });
  }
  return frames.filter((_, i) => keep[i]);
}

function fitKind(points, kind, lambda, centre) {
  const X = [], Y = [], w = [];
  points.forEach((p) => p.frames.forEach((f) => { X.push(MAPPINGS[kind](f, centre)); Y.push([p.x, p.y]); w.push(1 / p.frames.length); }));
  return { kind, centre, lambda, reg: fit(X, Y, lambda, w), n: X.length };
}

/** Error of the median prediction at each point when that point is left out of the fit. */
function lopo(points, kind, lambda, centre) {
  return points.map((p, i) => {
    const m = fitKind(points.filter((_, j) => j !== i), kind, lambda, centre);
    const pr = p.frames.map((f) => predict(m, f));
    return Math.hypot(median(pr.map((q) => q[0])) - p.x, median(pr.map((q) => q[1])) - p.y);
  });
}

/** points: [{x, y, frames: [features]}] in screen pixels. Returns the chosen model plus both candidates and their
    leave-one-point-out (LOPO) errors. The mapping and lambda are chosen by LOPO on calibration data only, so the
    held-out validation test stays an unbiased estimate. */
export function calibrate(points, opts = {}) {
  const minFrames = opts.minFrames || 8;
  let pts = points.map((p) => ({ ...p, frames: inliers(p.frames.filter(Boolean)) })).filter((p) => p.frames.length >= minFrames);
  if (pts.length < 6) throw new Error(`Only ${pts.length} calibration points had enough clean frames (need 6).`);
  const all = pts.flatMap((p) => p.frames);
  const centre = [0, 1, 2].map((j) => median(all.map((f) => f.e[j])));
  const kinds = opts.kinds || ["poly2", "v0"];
  const result = { models: {}, lopo_px: {}, dropped: [] };
  // One outlier point (a missed dot) is dropped if its LOPO error is > 2.5x the median of the others.
  const probe = lopo(pts, "poly2", 0.01, centre), med = median(probe);
  const worst = probe.indexOf(Math.max(...probe));
  if (pts.length > 7 && probe[worst] > 2.5 * med && probe[worst] > 60) {
    result.dropped.push({ x: pts[worst].x, y: pts[worst].y, lopo_px: probe[worst] });
    pts = pts.filter((_, i) => i !== worst);
  }
  for (const kind of kinds) {
    let best = null;
    const lams = kind === "v0" ? [1.0 / pts.reduce((s, p) => s + p.frames.length, 0)] : LAMBDAS;  // v0: its original lambda = 1 on unweighted frames
    for (const lam of lams) {
      const errs = lopo(pts, kind, lam, centre), e = mean(errs);
      if (!best || e < best.e) best = { lam, e, errs };
    }
    result.models[kind] = fitKind(pts, kind, best.lam, centre);
    result.lopo_px[kind] = best.e;
  }
  result.kind = kinds.reduce((a, b) => (result.lopo_px[b] < result.lopo_px[a] ? b : a));
  result.model = result.models[result.kind];
  result.open_median = centre[2];
  result.points = pts.length;
  result.frames = pts.reduce((s, p) => s + p.frames.length, 0);
  result.n = result.frames;
  return result;
}

/** Predict a screen point from features. Accepts a fitted mapping or a calibrate() result. */
export function predict(m, f) {
  if (m && m.model) m = m.model;
  return predictRaw(m.reg, MAPPINGS[m.kind](f, m.centre));
}

/* ---------- units ---------- */
export function deg2px(deg, pxPerCm, distCm) { return Math.tan((deg * Math.PI) / 180) * distCm * pxPerCm; }
export function px2deg(px, pxPerCm, distCm) { return (Math.atan(px / pxPerCm / distCm) * 180) / Math.PI; }

/* ---------- One Euro filter (Casiez et al., 2012) ---------- */
export class OneEuro {
  constructor(minCutoff = 1.0, beta = 0.01, dCutoff = 1) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: null, t: null }); }
  alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  reset() { this.x = null; this.dx = null; this.t = null; }
  filter(v, t) {
    if (this.x === null) { this.x = [...v]; this.dx = v.map(() => 0); this.t = t; return [...v]; }
    const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
    const ad = this.alpha(this.dCutoff, dt);
    this.dx = v.map((x, i) => ad * ((x - this.x[i]) / dt) + (1 - ad) * this.dx[i]);
    const speed = Math.hypot(...this.dx);                                 // one cutoff for both axes keeps the path straight
    const a = this.alpha(this.minCutoff + this.beta * speed, dt);
    this.x = v.map((x, i) => a * x + (1 - a) * this.x[i]);
    return [...this.x];
  }
}

/* ---------- fixation detector (online dispersion threshold, I-DT) ---------- */
export class Fixation {
  /** dispersionPx: max (x range + y range) inside a fixation; minMs: minimum duration. */
  constructor(dispersionPx = 120, minMs = 150) { Object.assign(this, { dispersionPx, minMs, buf: [] }); }
  reset() { this.buf = []; }
  push(x, y, t) {
    this.buf.push({ x, y, t });
    const disp = () => {
      const xs = this.buf.map((p) => p.x), ys = this.buf.map((p) => p.y);
      return Math.max(...xs) - Math.min(...xs) + Math.max(...ys) - Math.min(...ys);
    };
    while (this.buf.length > 1 && disp() > this.dispersionPx) this.buf.shift();
    const dur = this.buf[this.buf.length - 1].t - this.buf[0].t;
    return { fixating: dur >= this.minMs, duration_ms: dur, x: mean(this.buf.map((p) => p.x)), y: mean(this.buf.map((p) => p.y)) };
  }
}

/* ---------- snapping to page elements ---------- */
function erf(x) {                                                        // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

/** Likelihood of gaze point (x, y) if the user looks somewhere (uniformly) inside rect r, with isotropic Gaussian
    gaze error sigma: the box-blurred Gaussian, in closed form. */
export function rectLikelihood(x, y, r, sigma) {
  const w = Math.max(1, r.width), h = Math.max(1, r.height);
  const px = Phi((r.left + w - x) / sigma) - Phi((r.left - x) / sigma);
  const py = Phi((r.top + h - y) / sigma) - Phi((r.top - y) / sigma);
  return (px * py) / (w * h);
}

/** Normalise a target list from the page model. Accepts {id|n, label, rect:{left, top, width, height}} or
    {x, y, w, h}. Coordinates must be in the same space as the gaze point (screen pixels in Nullpoint). */
export function normTargets(list) {
  return (list || []).map((t, i) => {
    const r = t.rect || t;
    const rect = { left: +(r.left ?? r.x), top: +(r.top ?? r.y), width: +(r.width ?? r.w), height: +(r.height ?? r.h) };
    return { id: t.id ?? t.n ?? i, label: String(t.label ?? ""), rect };
  }).filter((t) => [t.rect.left, t.rect.top, t.rect.width, t.rect.height].every(Number.isFinite) && t.rect.width > 0 && t.rect.height > 0);
}

/** Posterior over page elements given one gaze point. A "none" hypothesis (gaze on nothing actionable, uniform over
    the screen) takes prior mass pNone so the highlight can go away. */
export function posterior(x, y, targets, sigma, screen = { width: 1920, height: 1080 }, pNone = 0.05) {
  const n = targets.length;
  if (!n) return { p_none: 1, items: [] };
  const prior = (1 - pNone) / n;
  const items = targets.map((t) => ({ id: t.id, label: t.label, s: prior * rectLikelihood(x, y, t.rect, sigma) }));
  const none = pNone / (screen.width * screen.height);
  const Z = items.reduce((s, it) => s + it.s, 0) + none;
  items.forEach((it) => { it.p = it.s / Z; delete it.s; });
  items.sort((a, b) => b.p - a.p);
  return { p_none: none / Z, items };
}

/** Stateful snapper: posterior + hysteresis + dwell. */
export class Snapper {
  constructor(opts = {}) {
    Object.assign(this, { sigma: 80, k: 3, pNone: 0.05, switchRatio: 1.5, switchMs: 120, dropMs: 300, dwellMs: 800,
      screen: { width: 1920, height: 1080 } }, opts);
    this.targets = []; this.reset();
  }
  reset() { this.cur = null; this.pending = null; this.dwellSince = null; this.dwelled = false; this.lastPost = null; this.gaze = null; this.t = 0; }
  setTargets(list) { this.targets = normTargets(list); if (this.cur !== null && !this.targets.some((t) => t.id === this.cur)) this.cur = null; }
  setSigma(px) { this.sigma = Math.max(5, px); }
  /** Feed one smoothed gaze point. fixating: from Fixation (dwell only counts during a fixation). */
  update(x, y, t, fixating = true) {
    const post = posterior(x, y, this.targets, this.sigma, this.screen, this.pNone);
    this.lastPost = post; this.gaze = { x, y }; this.t = t;
    const pOf = (id) => (id === null ? post.p_none : (post.items.find((it) => it.id === id) || { p: 0 }).p);
    const best = post.items.length && post.items[0].p > post.p_none ? post.items[0].id : null;
    const pc = pOf(this.cur), pb = pOf(best);
    let want = this.cur;
    if (best !== this.cur && pb >= this.switchRatio * pc) want = best;    // hysteresis: the challenger must clearly win
    if (want !== this.cur) {
      if (!this.pending || this.pending.id !== want) this.pending = { id: want, since: t };
      const need = want === null ? this.dropMs : this.switchMs;           // ...and keep winning for a while
      if (t - this.pending.since >= need) { this.cur = want; this.pending = null; this.dwellSince = null; this.dwelled = false; }
    } else this.pending = null;
    let dwell = false;
    if (this.cur !== null && fixating) {
      if (this.dwellSince === null) this.dwellSince = t;
      if (!this.dwelled && t - this.dwellSince >= this.dwellMs) { this.dwelled = true; dwell = true; }
    } else if (!fixating) { this.dwellSince = null; }
    return { highlight: this.cur, dwell, dwell_progress: this.dwellSince === null ? 0 : Math.min(1, (t - this.dwellSince) / this.dwellMs) };
  }
  /** Top-k candidates as a plain JSON structure for the voice decoder. */
  candidates(k = this.k) {
    const post = this.lastPost || { p_none: 1, items: [] };
    return {
      t: this.t, gaze: this.gaze, sigma_px: this.sigma, highlight: this.cur, p_none: post.p_none,
      candidates: post.items.slice(0, k).map((it) => ({ id: it.id, label: it.label, p: +it.p.toFixed(4) })),
    };
  }
}

/** Fuse gaze candidates with voice word probabilities: p(element | gaze, voice) ∝ p_gaze(element) · p_voice(label).
    voice: {word: probability}; labels not heard get `floor`. Gaze narrows which elements, voice picks the word. */
export function fuseVoice(cands, voice, floor = 1e-3) {
  const list = (cands.candidates || cands).map((c) => ({ ...c, s: c.p * (voice[String(c.label).toLowerCase()] ?? floor) }));
  const Z = list.reduce((s, c) => s + c.s, 0) || 1;
  return list.map(({ s, ...c }) => ({ ...c, p: s / Z })).sort((a, b) => b.p - a.p);
}

/* ---------- camera + landmarker ---------- */
export async function start(videoEl, status) {
  const { FaceLandmarker, FilesetResolver } = await import("/vendor/mediapipe/vision_bundle.mjs");
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
  running = true; fpsT = now(); loop();
}
export function stop() {
  running = false;
  if (video && video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
}
function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    const t = now();
    const r = landmarker.detectForVideo(video, t);
    const lm = r.faceLandmarks && r.faceLandmarks[0];
    const aspect = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 1;
    const raw = lm ? features(lm, r.facialTransformationMatrixes && r.facialTransformationMatrixes[0], aspect, 0) : null;
    const f = raw && raw.e[2] >= blinkThr ? raw : null;
    last = { f, t, face: !!lm, open_ratio: raw ? raw.e[2] : null };
    stat.frames++; if (lm) stat.face++; if (f) stat.eyes++;
    frames++;
    if (t - fpsT > 1000) { fps = frames; frames = 0; fpsT = t; }
    if (onFrame) onFrame(last);
  }
  requestAnimationFrame(loop);
}
export function info() { return { fps, face: !!(last && last.face), open: !!(last && last.f), calibrated: !!model }; }
export function setModel(m) { model = m; }
/* Blink cut-off on the lid gap (eye widths). Fixed values drop open eyes for some people and when looking down,
   so it starts low and is set from each person's calibration (60% of their median open-eye gap). */
let blinkThr = 0.08;
export function setBlinkThreshold(x) { blinkThr = Math.max(0.05, Math.min(0.12, x)); }
export function getBlinkThreshold() { return blinkThr; }
/* Share of frames with a face and with usable (open) eyes since resetStats(), logged with every test. */
let stat = { frames: 0, face: 0, eyes: 0 };
export function resetStats() { stat = { frames: 0, face: 0, eyes: 0 }; }
export function stats() { const n = Math.max(1, stat.frames); return { frames: stat.frames, face_rate: stat.face / n, eyes_rate: stat.eyes / n }; }
export function getModel() { return model; }
export function subscribe(fn) { onFrame = fn; }

/** Collect features for `ms` milliseconds after `settle` ms of settling time. */
export function collect(ms, settle = 600) {
  return new Promise((resolve) => {
    const out = [], t0 = now();
    const prev = onFrame;
    onFrame = (fr) => {
      const t = now() - t0;
      if (t > settle && fr.f) out.push({ ...fr.f, t: fr.t });
      if (t > settle + ms) { onFrame = prev; resolve(out); }
      if (prev) prev(fr);
    };
  });
}

/** Accuracy (distance of the median estimate to the target), precision (RMS sample-to-sample) and sigma, the
    per-axis SD of an isotropic Gaussian fitted to all (smoothed) estimates: sigma = sqrt(mean(|e|^2) / 2).
    points: [{x, y, preds: [[x, y]...], smooth?: [[x, y]...]}]. */
export function score(points, pxPerCm, distCm) {
  const deg = (px) => px2deg(px, pxPerCm, distCm);
  const acc = [], prec = [], sq = [];
  points.forEach((p) => {
    if (!p.preds.length) return;
    const mx = median(p.preds.map((q) => q[0])), my = median(p.preds.map((q) => q[1]));
    acc.push(Math.hypot(mx - p.x, my - p.y));
    let s = 0; for (let i = 1; i < p.preds.length; i++) s += (p.preds[i][0] - p.preds[i - 1][0]) ** 2 + (p.preds[i][1] - p.preds[i - 1][1]) ** 2;
    prec.push(Math.sqrt(s / Math.max(1, p.preds.length - 1)));
    (p.smooth || p.preds).forEach((q) => sq.push((q[0] - p.x) ** 2 + (q[1] - p.y) ** 2));
  });
  const sigma = Math.sqrt(mean(sq) / 2);
  return { points: acc.length, accuracy_px: mean(acc), precision_px: mean(prec), accuracy_deg: deg(mean(acc)),
    precision_deg: deg(mean(prec)), sigma_px: sigma, sigma_deg: deg(sigma), per_point_px: acc };
}
