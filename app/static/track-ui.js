/* Track tab: camera, 9-point calibration, accuracy test, 12-target selection test, live dot. */
import * as T from "/static/track.js";

const $ = (id) => document.getElementById(id);
const stage = $("stage"), help = $("stage-help");
let live = false, filter = null, gazeEl = null;

function say(t, err) { NP.msg("track-msg", t, err); }
function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }
function enable(on) { ["t-cal"].forEach((id) => ($(id).disabled = !on)); ["t-val", "t-grid", "t-live"].forEach((id) => ($(id).disabled = !(on && T.getModel()))); }

setInterval(() => {
  const i = T.info();
  $("t-fps").textContent = i.fps; $("t-face").textContent = i.face ? (i.open ? "yes" : "blink") : "no";
}, 500);

$("cam-start").addEventListener("click", async () => {
  try { await T.start($("cam"), say); say("Camera on. Look at the screen, then press Calibrate."); enable(true); }
  catch (e) { say("Could not start the camera: " + e.message, true); }
});
$("cam-stop").addEventListener("click", () => { T.stop(); enable(false); say("Camera off."); });

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
  return d;
}

$("t-cal").addEventListener("click", async () => {
  await openStage();
  help.textContent = "Look at each orange dot until it shrinks.";
  const X = [], Y = [];
  const pts = []; for (const fy of [0.1, 0.5, 0.9]) for (const fx of [0.1, 0.5, 0.9]) pts.push([fx, fy]);
  for (const [fx, fy] of pts) {
    const d = dotAt(fx, fy);
    const feats = await T.collect(900, 500);
    feats.forEach((f) => { X.push(f); Y.push([fx * innerWidth, fy * innerHeight]); });
    d.remove();
  }
  closeStage();
  if (X.length < 60) { say(`Only ${X.length} usable frames. Make sure your face is lit and visible, then try again.`, true); return; }
  T.setModel(T.fit(X, Y, 1.0));
  filter = new T.OneEuro();
  enable(true);
  say(`Calibrated with ${X.length} frames. Run the accuracy test next.`);
});

$("t-val").addEventListener("click", async () => {
  await openStage();
  help.textContent = "Look at each dot until it disappears.";
  const pts = [[0.3, 0.3], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.7, 0.7]].map(([fx, fy]) => ({ x: fx * innerWidth, y: fy * innerHeight, fx, fy, preds: [] }));
  for (const p of pts) {
    const d = dotAt(p.fx, p.fy);
    const feats = await T.collect(1000, 600);
    p.preds = feats.map((f) => T.predict(T.getModel(), f));
    d.remove();
  }
  closeStage();
  const pxPerCm = screen.width / Math.max(1, +$("scr-cm").value), dist = +$("dist-cm").value || 60;
  const s = T.score(pts, pxPerCm, dist);
  $("t-result").textContent = `Accuracy ${s.accuracy_deg.toFixed(2)}° (${Math.round(s.accuracy_px)} px), precision ${s.precision_deg.toFixed(2)}° (${Math.round(s.precision_px)} px), ${s.points} points.`;
  NP.post("/api/track/result", { kind: "validation", profile: NP.prof(), ...s, screen_px: [screen.width, screen.height], screen_cm: +$("scr-cm").value,
    distance_cm: dist, calibration_samples: T.getModel().n }).then(() => NP.refreshStatus()).catch(() => {});
});

$("t-grid").addEventListener("click", async () => {
  await openStage();
  const grid = document.createElement("div"); grid.className = "grid";
  for (let i = 1; i <= 12; i++) { const c = document.createElement("div"); c.textContent = i; grid.appendChild(c); }
  stage.appendChild(grid);
  const cells = [...grid.children];
  const f = new T.OneEuro();
  let hot = -1;
  T.subscribe((fr) => {
    if (!fr.f) return;
    const [x, y] = f.filter(T.predict(T.getModel(), fr.f), fr.t);
    let best = -1, bd = 1e9;
    cells.forEach((c, i) => { const r = c.getBoundingClientRect(); const d = Math.hypot(x - (r.left + r.width / 2), y - (r.top + r.height / 2)); if (d < bd) { bd = d; best = i; } });
    if (best !== hot) { cells.forEach((c, i) => c.classList.toggle("hot", i === best)); hot = best; }
  });
  const order = [...Array(12).keys()].sort(() => Math.random() - 0.5);
  let hits = 0; const times = [];
  for (const goal of order) {
    cells.forEach((c, i) => c.classList.toggle("goal", i === goal));
    help.textContent = `Look at the orange square (${goal + 1}) and press Space when it turns teal. ${times.length + 1} of 12.`;
    const t0 = performance.now();
    await new Promise((res) => {
      const k = (e) => { if (e.code === "Space") { e.preventDefault(); removeEventListener("keydown", k); res(); } };
      addEventListener("keydown", k);
    });
    times.push(performance.now() - t0); if (hot === goal) hits++;
  }
  T.subscribe(null); closeStage();
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  $("t-result").textContent = `Target test: ${hits} of 12 correct (${Math.round((100 * hits) / 12)}%), ${Math.round(mean)} ms per selection.`;
  NP.post("/api/track/result", { kind: "target_test", profile: NP.prof(), hit_rate: hits / 12, trials: 12, mean_time_ms: mean,
    screen_px: [screen.width, screen.height] }).then(() => NP.refreshStatus()).catch(() => {});
});

$("t-live").addEventListener("click", () => {
  live = !live; $("t-live").textContent = live ? "Stop live dot" : "Live gaze dot";
  if (!live) { T.subscribe(null); if (gazeEl) gazeEl.remove(); gazeEl = null; return; }
  gazeEl = document.createElement("div");
  gazeEl.style.cssText = "position:fixed;width:24px;height:24px;margin:-12px 0 0 -12px;border-radius:50%;border:3px solid var(--signal);pointer-events:none;z-index:60;transition:left .03s,top .03s";
  document.body.appendChild(gazeEl);
  filter = new T.OneEuro();
  T.subscribe((fr) => {
    if (!fr.f || !T.getModel()) return;
    const [x, y] = filter.filter(T.predict(T.getModel(), fr.f), fr.t);
    gazeEl.style.left = Math.max(0, Math.min(innerWidth, x)) + "px"; gazeEl.style.top = Math.max(0, Math.min(innerHeight, y)) + "px";
  });
});
