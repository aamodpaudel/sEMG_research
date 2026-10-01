/* Track tab: camera, 9-point calibration, held-out accuracy test (compares mappings, estimates sigma),
   12-target selection test (snap vs pixel hit, measured on the same trials), and live snapping to page elements
   with top-k candidates for the voice decoder. All gaze coordinates are screen pixels (CSS px). */
import * as T from "/static/track.js";

const $ = (id) => document.getElementById(id);
const stage = $("stage"), help = $("stage-help");
let live = false, gazeEl = null, calib = null, sigmaSource = "default";
let externalTargets = null, liveTimer = null, lastPost = 0, hotEl = null;
const snapper = new T.Snapper({ k: 3 });
const fixation = new T.Fixation();

function say(t, err) { NP.msg("track-msg", t, err); }
function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }
function geom() { return { pxPerCm: screen.width / Math.max(1, +$("scr-cm").value || 34.5), dist: +$("dist-cm").value || 60 }; }
function degOf(px) { const g = geom(); return T.px2deg(px, g.pxPerCm, g.dist); }
function pxOf(deg) { const g = geom(); return T.deg2px(deg, g.pxPerCm, g.dist); }
function enable(on) {
  $("t-cal").disabled = !on;
  ["t-val", "t-grid", "t-live"].forEach((id) => ($(id).disabled = !(on && T.getModel())));
}

/* Screen <-> viewport. In fullscreen they are the same; windowed, the offset is estimated from the window frame. */
function offset() {
  if (document.fullscreenElement) return { x: 0, y: 0 };
  const border = Math.max(0, (outerWidth - innerWidth) / 2);
  return { x: screenX + border, y: screenY + Math.max(0, outerHeight - innerHeight - border) };
}
function toScreenRect(r) { const o = offset(); return { left: r.left + o.x, top: r.top + o.y, width: r.width, height: r.height }; }
function toViewport(x, y) { const o = offset(); return [x - o.x, y - o.y]; }

setInterval(() => {
  const i = T.info();
  $("t-fps").textContent = i.fps; $("t-face").textContent = i.face ? (i.open ? "yes" : "blink") : "no";
}, 500);

$("cam-start").addEventListener("click", async () => {
  try { await T.start($("cam"), say); say("Camera on. Look at the screen, then press Calibrate."); enable(true); }
  catch (e) { say("Could not start the camera: " + e.message, true); }
});
$("cam-stop").addEventListener("click", () => { T.stop(); stopLive(); enable(false); say("Camera off."); });

/* While a test runs full screen, show whether the camera sees a face and open eyes (otherwise a test can fail
   silently: no frames means no gaze estimate, and nothing snaps). */
const chip = document.createElement("div"); chip.className = "track-chip"; stage.appendChild(chip);
setInterval(() => {
  if (stage.hidden) return;
  const i = T.info();
  chip.textContent = !i.face ? "Face not found: sit in front of the camera" : !i.open ? "Eyes not seen: more light on your face, look at the screen" : "Face and eyes tracked";
  chip.dataset.ok = i.face && i.open ? "1" : "0";
}, 200);

async function openStage() {
  stage.hidden = false; stage.querySelectorAll(".dot,.grid,.gaze").forEach((n) => n.remove());
  try { await document.documentElement.requestFullscreen(); } catch (e) { /* windowed is fine */ }
  await wait(400);
}
function closeStage() { stage.hidden = true; if (document.fullscreenElement) document.exitFullscreen().catch(() => {}); }
function dotAt(fx, fy) {
  const d = document.createElement("div"); d.className = "dot";
  d.style.left = fx * innerWidth + "px"; d.style.top = fy * innerHeight + "px";
  stage.appendChild(d); requestAnimationFrame(() => d.classList.add("shrink"));
  const o = offset();
  return { el: d, x: fx * innerWidth + o.x, y: fy * innerHeight + o.y };
}
function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

/* ---------- 9-point calibration: 3x3 grid, 0.6 s settling, 1 s of frames, outlier frames rejected ---------- */
$("t-cal").addEventListener("click", async () => {
  stopLive();
  await openStage();
  help.textContent = "Look at each orange dot until it disappears. Keep your head natural; small movements are fine.";
  const grid = []; for (const fy of [0.1, 0.5, 0.9]) for (const fx of [0.1, 0.5, 0.9]) grid.push([fx, fy]);
  const points = [];
  for (const [fx, fy] of shuffle(grid)) {
    let frames = [];
    for (let attempt = 0; attempt < 2; attempt++) {                    // one retry if the point was mostly blinks
      const d = dotAt(fx, fy);
      frames = await T.collect(1000, 600);
      d.el.remove();
      if (T.inliers(frames).length >= 10) { points.push({ x: d.x, y: d.y, frames }); break; }
      help.textContent = "Once more: look at the dot.";
      await wait(300);
    }
  }
  closeStage();
  try { calib = T.calibrate(points); }
  catch (e) { say(e.message + " Make sure your face is lit and visible, then try again.", true); return; }
  T.setModel(calib);
  T.setBlinkThreshold(0.6 * calib.open_median);
  const lopo = calib.lopo_px[calib.kind];
  snapper.setSigma(Math.max(pxOf(0.5), lopo / Math.sqrt(Math.PI / 2))); sigmaSource = "calibration (leave-one-point-out)";
  enable(true);
  const cmp = Object.entries(calib.lopo_px).map(([k, v]) => `${k} ${degOf(v).toFixed(2)}°`).join(", ");
  say(`Calibrated: ${calib.points} points, ${calib.frames} frames${calib.dropped.length ? `, 1 point dropped as an outlier` : ""}. ` +
      `Leave-one-point-out error: ${cmp}. Using ${calib.kind}. Run the accuracy test next.`);
});

/* ---------- accuracy test: 5 points not used in calibration; scores every mapping on the same frames ---------- */
$("t-val").addEventListener("click", async () => {
  stopLive();
  await openStage();
  help.textContent = "Look at each dot until it disappears.";
  T.resetStats();
  const spots = [[0.3, 0.3], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.7, 0.7]];
  const pts = [];
  for (const [fx, fy] of spots) {
    const d = dotAt(fx, fy);
    pts.push({ x: d.x, y: d.y, frames: await T.collect(1000, 600) });
    d.el.remove();
  }
  closeStage();
  const g = geom(), by = {};
  for (const [kind, m] of Object.entries(calib.models)) {
    const scored = pts.map((p) => {
      const f = new T.OneEuro();
      const preds = p.frames.map((fr) => T.predict(m, fr));
      return { x: p.x, y: p.y, preds, smooth: preds.map((q, i) => f.filter(q, p.frames[i].t)) };
    });
    by[kind] = T.score(scored, g.pxPerCm, g.dist);
  }
  const s = by[calib.kind];
  const st = T.stats();
  if (!s.points) { say(`No usable frames in the accuracy test (face seen ${Math.round(100 * st.face_rate)}% of the time, open eyes ${Math.round(100 * st.eyes_rate)}%). Add front light, sit at arm's length, then calibrate again.`, true); return; }
  snapper.setSigma(Math.max(pxOf(0.5), s.sigma_px)); sigmaSource = "accuracy test";
  const other = Object.entries(by).filter(([k]) => k !== calib.kind).map(([k, v]) => `${k} ${v.accuracy_deg.toFixed(2)}°`).join(", ");
  $("t-result").textContent = `Accuracy ${s.accuracy_deg.toFixed(2)}° (${Math.round(s.accuracy_px)} px), precision ${s.precision_deg.toFixed(2)}°, ` +
    `snap σ ${s.sigma_deg.toFixed(2)}°, ${s.points} of 5 points, mapping ${calib.kind}. Other mapping on the same frames: ${other}. ` +
    `Eyes tracked ${Math.round(100 * st.eyes_rate)}% of the time.` + (s.points < 4 ? " Some dots had no usable frames: check the light and calibrate again." : "") +
    (s.accuracy_deg > 5 ? " The error is large: check lighting and distance, then calibrate again." : "");
  const extra = {};
  for (const [k, v] of Object.entries(by)) { extra[`accuracy_deg_${k}`] = v.accuracy_deg; extra[`accuracy_px_${k}`] = v.accuracy_px; }
  NP.post("/api/track/result", { kind: "validation", profile: NP.prof(), ...s, ...extra, ...T.stats(), blink_threshold: T.getBlinkThreshold(), mapping: calib.kind,
    lopo_deg: degOf(calib.lopo_px[calib.kind]), dropped_points: calib.dropped.length,
    screen_px: [screen.width, screen.height], screen_cm: +$("scr-cm").value, distance_cm: g.dist,
    calibration_samples: calib.frames }).then(() => NP.refreshStatus()).catch(() => {});
});

/* ---------- target test: 12 cells; snap hit and pixel hit on the same trials ---------- */
$("t-grid").addEventListener("click", async () => {
  stopLive();
  await openStage();
  const grid = document.createElement("div"); grid.className = "grid";
  for (let i = 1; i <= 12; i++) { const c = document.createElement("div"); c.textContent = i; grid.appendChild(c); }
  stage.appendChild(grid);
  await wait(50);
  const cells = [...grid.children];
  const rects = cells.map((c) => toScreenRect(c.getBoundingClientRect()));
  snapper.reset(); snapper.setTargets(rects.map((r, i) => ({ id: i, label: String(i + 1), rect: r })));
  snapper.screen = { width: screen.width, height: screen.height };
  const f = new T.OneEuro(); fixation.reset(); T.resetStats();
  const dot = document.createElement("div"); dot.className = "gaze"; dot.hidden = true; stage.appendChild(dot);
  let hot = null, gaze = null;
  T.subscribe((fr) => {
    if (!fr.f) return;
    const [x, y] = f.filter(T.predict(T.getModel(), fr.f), fr.t);
    gaze = [x, y];
    const [vx, vy] = toViewport(x, y);                       // faint ring: where the model thinks you look
    dot.hidden = false; dot.style.left = Math.max(0, Math.min(innerWidth, vx)) + "px"; dot.style.top = Math.max(0, Math.min(innerHeight, vy)) + "px";
    const st = snapper.update(x, y, fr.t, fixation.push(x, y, fr.t).fixating);
    if (st.highlight !== hot) { cells.forEach((c, i) => c.classList.toggle("hot", i === st.highlight)); hot = st.highlight; }
  });
  const order = shuffle([...Array(12).keys()]);
  let snapHits = 0, pixelHits = 0, centreHits = 0, skipped = 0; const times = [];
  for (const goal of order) {
    cells.forEach((c, i) => c.classList.toggle("goal", i === goal));
    help.textContent = `Look at the orange square (${goal + 1}) and press Space when it turns teal. ${times.length + 1} of 12. ` +
      "If it never turns teal, it moves on after 8 s.";
    const t0 = performance.now();
    const timedOut = await new Promise((res) => {
      const done = (v) => { removeEventListener("keydown", k); clearTimeout(to); res(v); };
      const k = (e) => { if (e.code === "Space") { e.preventDefault(); done(false); } };
      const to = setTimeout(() => done(hot !== goal), 8000);        // skip, counted as a miss unless it is teal now
      addEventListener("keydown", k);
    });
    if (timedOut) skipped++;
    times.push(performance.now() - t0);
    if (hot === goal) snapHits++;
    if (gaze) {
      const r = rects[goal];
      if (gaze[0] >= r.left && gaze[0] <= r.left + r.width && gaze[1] >= r.top && gaze[1] <= r.top + r.height) pixelHits++;
      const cd = rects.map((q) => Math.hypot(gaze[0] - q.left - q.width / 2, gaze[1] - q.top - q.height / 2));
      if (cd.indexOf(Math.min(...cd)) === goal) centreHits++;
    }
  }
  T.subscribe(null); closeStage();
  const mean = times.reduce((a, b) => a + b, 0) / times.length, pct = (n) => Math.round((100 * n) / 12);
  $("t-result").textContent = `Target test: snap ${snapHits}/12 (${pct(snapHits)}%), pixel cursor ${pixelHits}/12 (${pct(pixelHits)}%) ` +
    `on the same trials, ${Math.round(mean)} ms per selection` + (skipped ? `, ${skipped} skipped after 8 s` : "") +
    `. Eyes tracked ${Math.round(100 * T.stats().eyes_rate)}% of the time.`;
  NP.post("/api/track/result", { kind: "target_test", profile: NP.prof(), ...T.stats(), skipped, blink_threshold: T.getBlinkThreshold(),
    hit_rate: snapHits / 12, pixel_hit_rate: pixelHits / 12,
    centre_hit_rate: centreHits / 12, trials: 12, mean_time_ms: mean, sigma_px: snapper.sigma, mapping: calib && calib.kind,
    screen_px: [screen.width, screen.height] }).then(() => NP.refreshStatus()).catch(() => {});
});

/* ---------- live snapping ---------- */
const PAGE_SELECTOR = "button, a[href], input, select, textarea, [role=tab], [role=button]";
function pageTargets() {
  if (externalTargets) return externalTargets;
  const out = [];
  document.querySelectorAll(PAGE_SELECTOR).forEach((el, i) => {
    if (el.disabled || el.closest("[hidden]")) return;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) return;
    if (!el.dataset.gz) el.dataset.gz = "g" + i;
    const lab = el.getAttribute("aria-label") || (el.labels && el.labels[0] ? el.labels[0].textContent : "") || el.textContent || el.value || el.id || "";
    out.push({ id: el.dataset.gz, label: lab.trim().replace(/\s+/g, " ").slice(0, 40), rect: toScreenRect(r) });
  });
  return out;
}
function setHot(id, dwell) {
  if (hotEl && (!id || hotEl.dataset.gz !== id)) { hotEl.classList.remove("gz-hot", "gz-dwell"); hotEl = null; }
  if (id && !externalTargets) { hotEl = document.querySelector(`[data-gz="${id}"]`); if (hotEl) hotEl.classList.add("gz-hot"); }
  if (dwell && hotEl) hotEl.classList.add("gz-dwell");
}
function stopLive() {
  if (!live) return;
  live = false; $("t-live").textContent = "Live snap";
  T.subscribe(null); clearInterval(liveTimer); setHot(null);
  if (gazeEl) gazeEl.remove(); gazeEl = null;
}
$("t-live").addEventListener("click", () => {
  if (live) { stopLive(); return; }
  live = true; $("t-live").textContent = "Stop live snap";
  gazeEl = document.createElement("div"); gazeEl.className = "gz-dot"; document.body.appendChild(gazeEl);
  const f = new T.OneEuro(); fixation.reset(); snapper.reset();
  snapper.screen = { width: screen.width, height: screen.height };
  snapper.setTargets(pageTargets());
  // the page model can change; when the browser extension is connected, snap to that page's targets instead
  liveTimer = setInterval(() => {
    NP.api("/api/context").then((c) => {
      const ext = (c.targets || []).filter((t) => t.rect && t.rect.width > 1 && t.rect.height > 1)
        .map((t) => ({ id: "n" + t.n, label: t.label, rect: t.rect }));
      externalTargets = ext.length ? T.normTargets(ext) : null;
    }).catch(() => {}).finally(() => snapper.setTargets(pageTargets()));
  }, 500);
  T.subscribe((fr) => {
    if (!fr.f || !T.getModel()) return;
    const [x, y] = f.filter(T.predict(T.getModel(), fr.f), fr.t);
    const fx = fixation.push(x, y, fr.t);
    const st = snapper.update(x, y, fr.t, fx.fixating);
    const [vx, vy] = toViewport(x, y);
    gazeEl.style.left = Math.max(0, Math.min(innerWidth, vx)) + "px"; gazeEl.style.top = Math.max(0, Math.min(innerHeight, vy)) + "px";
    setHot(st.highlight, st.dwell);
    if (fr.t - lastPost > 200) {                                           // 5 Hz: panel + server for the decoder
      lastPost = fr.t;
      const c = { ...snapper.candidates(10), fixating: fx.fixating, sigma_source: sigmaSource };  // decoder fuses softly over these
      $("t-cands").textContent = JSON.stringify(c, null, 1);
      NP.post("/api/track/candidates", c).catch(() => {});
    }
  });
});

/* Interface for the page model and the voice decoder:
   AfonoTrack.setTargets([{id, label, rect: {left, top, width, height}}])  rects in screen CSS px (null = this page)
   AfonoTrack.candidates(k) -> {t, gaze, sigma_px, highlight, p_none, candidates: [{id, label, p}]}
   AfonoTrack.fuse(voice) -> candidates re-ranked by p_gaze * p_voice(label), voice = {word: probability} */
window.AfonoTrack = {
  setTargets(list) { externalTargets = list ? T.normTargets(list) : null; snapper.setTargets(pageTargets()); },
  candidates: (k) => snapper.candidates(k),
  fuse: (voice, k) => T.fuseVoice(snapper.candidates(k), voice),
  snapper, T,
};
