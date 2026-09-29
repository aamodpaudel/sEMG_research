/* Afono site: live numbers, motion, and the hero signal. */
(function () {
  "use strict";
  var root = document.documentElement;
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var P = window.AFONO_PROGRESS || null;

  function fmt(v) { return (Math.round(v * 10) / 10).toString(); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  /* ---------- live numbers from the run log ---------- */
  if (P) {
    var h = P.headline || {};
    var vals = {
      dates_word_acc: h.dates_word_acc, closed100_word_acc: h.closed100_word_acc, closed500_word_acc: h.closed500_word_acc,
      open_wer: h.open_wer, open_acc: h.open_wer == null ? null : 100 - h.open_wer, measured: h.measured,
      model_mb: P.demo && P.demo.model_mb, updated: (P.updated || "").slice(0, 10)
    };
    var T = {};
    ((P.research || {}).tasks || []).forEach(function (t) { T[t.task] = t; });
    function ci(t) { return t && t.acc_ci ? fmt(t.acc_ci[0]) + "–" + fmt(t.acc_ci[1]) + "%" : null; }
    vals.dates_ci = ci(T.dates); vals.closed500_ci = ci(T.closed500); vals.closed100_ci = ci(T.closed100);
    vals.dates_itr = T.dates ? Math.round(T.dates.itr) : null;
    vals.dates_cwpm = T.dates ? Math.round(T.dates.cwpm) : null;
    window.AFONO_TASKS = T;
    document.querySelectorAll("[data-p]").forEach(function (e) {
      var v = vals[e.dataset.p];
      if (v != null) e.textContent = typeof v === "number" ? fmt(v) : v;
    });
    document.querySelectorAll("[data-w]").forEach(function (e) {
      var v = vals[e.dataset.w];
      if (v != null) e.style.width = Math.max(0, Math.min(100, v)) + "%";
    });
    var log = document.getElementById("runlog");
    if (log && P.runs && P.runs.length) {
      log.textContent = "";
      P.runs.forEach(function (r) {
        var li = el("li", r.kind);
        li.appendChild(el("span", "meta", r.date + " · " + r.kind + " · " + r.run_id));
        li.appendChild(el("span", "sum", r.summary));
        log.appendChild(li);
      });
    }
  }

  /* ---------- headline: words blur in ---------- */
  var hl = document.getElementById("headline");
  if (hl && !reduce) {
    var words = hl.textContent.trim().split(/\s+/);
    hl.setAttribute("aria-label", hl.textContent.trim());
    hl.textContent = "";
    words.forEach(function (w, i) {
      var s = el("span", "w", w); s.setAttribute("aria-hidden", "true");
      s.style.animationDelay = (0.15 + i * 0.06) + "s";
      hl.appendChild(s);
      if (i < words.length - 1) hl.appendChild(document.createTextNode(" "));
    });
  }

  /* ---------- header frosts once you scroll ---------- */
  var bar = document.getElementById("top-bar");
  if (bar) {
    var onScroll = function () { bar.classList.toggle("scrolled", window.scrollY > 8); };
    window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  }

  /* ---------- scroll reveals (content stays visible if this script never runs) ---------- */
  function countUp(e) {
    var target = parseFloat(e.textContent), dec = (e.textContent.split(".")[1] || "").length;
    if (!isFinite(target) || reduce) return;
    var t0 = null, dur = 1200;
    (function step(t) {
      if (t0 === null) t0 = t;
      var k = Math.min(1, (t - t0) / dur), v = target * (1 - Math.pow(1 - k, 3));
      e.textContent = v.toFixed(dec);
      if (k < 1) requestAnimationFrame(step); else e.textContent = target.toFixed(dec);
    })(performance.now());
  }
  function reveal(n) {
    n.classList.add("in");
    if (n.id === "flow") n.classList.remove("pre");
    n.querySelectorAll("[data-count]").forEach(countUp);
  }
  var rvs = [].slice.call(document.querySelectorAll(".rv"));
  var flow = document.getElementById("flow");
  if ("IntersectionObserver" in window && !reduce) {
    var below = function (n) { return n.getBoundingClientRect().top > window.innerHeight * 0.92; };
    root.classList.add("js-rv");
    if (flow && below(flow)) flow.classList.add("pre");
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { reveal(e.target); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    rvs.forEach(function (n) { if (below(n)) io.observe(n); else n.classList.add("in"); });
    if (flow) io.observe(flow);
  } else {
    rvs.forEach(function (n) { n.classList.add("in"); });
  }

  /* ---------- vision line types itself ---------- */
  var vision = document.getElementById("vision");
  if (vision && !reduce) {
    var html = vision.innerHTML, text = vision.textContent;
    vision.setAttribute("aria-label", text);
    setTimeout(function () {
      var i = 0, caret = el("span", "caret"); caret.setAttribute("aria-hidden", "true");
      var span = el("span"); span.setAttribute("aria-hidden", "true");
      vision.textContent = ""; vision.appendChild(span); vision.appendChild(caret);
      (function tick() {
        i += 1; span.textContent = text.slice(0, i);
        if (i < text.length) setTimeout(tick, 22 + Math.random() * 30);
        else setTimeout(function () { vision.innerHTML = html; vision.appendChild(caret); setTimeout(function () { caret.remove(); }, 2400); }, 250);
      })();
    }, 2700);
  }

  /* ---------- small interactions ---------- */
  var intro = document.getElementById("intro"), replay = document.getElementById("replay");
  if (intro && replay) replay.addEventListener("click", function () {
    intro.classList.remove("intro"); void intro.offsetWidth; intro.classList.add("intro");
  });
  var copy = document.getElementById("copy-email");
  if (copy) copy.addEventListener("click", function () {
    var addr = document.querySelector(".contact .addr");
    var done = function () { copy.textContent = "Copied"; setTimeout(function () { copy.textContent = "Copy email"; }, 1600); };
    try { navigator.clipboard.writeText(addr.textContent.trim()).then(done, function () { selectText(addr); }); }
    catch (e) { selectText(addr); }
  });
  function selectText(node) {
    var r = document.createRange(); r.selectNodeContents(node);
    var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
  }

  /* ---------- shared loader for the demo recordings ---------- */
  var FILES = window.AFONO_FILES || {};
  var samplesP = null;
  window.afonoSamples = function () {
    if (!samplesP) samplesP = Promise.all([
      fetch("demo/samples.json").then(function (r) { if (!r.ok) throw new Error("samples.json " + r.status); return r.json(); }),
      fetch(FILES.samples || "demo/samples.bin").then(function (r) { if (!r.ok) throw new Error("samples.bin " + r.status); return r.arrayBuffer(); })
    ]).then(function (x) { return { info: x[0], data: new Float32Array(x[1]) }; });
    return samplesP;
  };

  /* ---------- hero signal: the real muscle-activity envelope, drawn smooth ----------
     The envelope is the RMS of all 8 channels over 25 ms windows, taken from the demo
     recordings. It swells when the speaker mouths words and goes flat between them. */
  var cv = document.getElementById("emg-bg");
  if (!cv || !cv.getContext) return;
  var ctx = cv.getContext("2d"), env = null, visible = true, t0 = performance.now(), dpr = 1;

  function buildEnvelope(s) {
    var out = [], WIN = 25;
    s.info.samples.forEach(function (m) {
      var C = m.channels, base = m.offset;
      for (var i = 0; i + WIN <= m.length; i += WIN) {
        var acc = 0;
        for (var k = 0; k < WIN; k++) for (var c = 0; c < C; c++) { var v = s.data[base + (i + k) * C + c]; acc += v * v; }
        out.push(Math.sqrt(acc / (WIN * C)));
      }
      for (var z = 0; z < 24; z++) out.push(0);            // a breath of silence between recordings
    });
    var sm = out.map(function (_, i) { var a = 0, n = 0; for (var j = -4; j <= 4; j++) { var v = out[i + j]; if (v != null) { a += v; n++; } } return a / n; });
    var sorted = sm.slice().sort(function (a, b) { return a - b; }), hi = sorted[Math.floor(sorted.length * 0.97)] || 1;
    return sm.map(function (v) { var x = Math.max(0, v / hi - 0.32) / 0.68; return Math.min(1, Math.pow(x, 1.25)); });
  }
  function size() {
    var r = cv.getBoundingClientRect(); dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.max(1, Math.round(r.width * dpr)); cv.height = Math.max(1, Math.round(r.height * dpr));
  }
  function envAt(p) {
    var n = env.length, i = Math.floor(p), f = p - i;
    var a = env[((i % n) + n) % n], b = env[(((i + 1) % n) + n) % n];
    return a + (b - a) * (f * f * (3 - 2 * f));
  }
  function draw(now) {
    if (!env) return;
    var W = cv.width, H = cv.height, mid = H / 2, t = (now - t0) / 1000;
    var teal = getComputedStyle(root).getPropertyValue("--teal").trim() || "#0F5E63";
    var orange = getComputedStyle(root).getPropertyValue("--signal").trim() || "#D9552B";
    var gap = 5 * dpr, shift = t * 70 * dpr, maxA = H * 0.42, step = 2 * dpr;
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = "source-over";
    ctx.beginPath();
    for (var x = 0; x <= W; x += step) ctx.lineTo(x, mid - envAt((x + shift) / gap) * maxA);
    for (x = W; x >= 0; x -= step) ctx.lineTo(x, mid + envAt((x + shift) / gap) * maxA);
    ctx.closePath(); ctx.globalAlpha = 0.12; ctx.fillStyle = teal; ctx.fill();
    ctx.beginPath(); ctx.globalAlpha = 0.95; ctx.strokeStyle = teal; ctx.lineWidth = 1.6 * dpr; ctx.lineJoin = "round";
    for (x = 0; x <= W; x += step) {
      var a = envAt((x + shift) / gap) * maxA;
      var y = mid + a * Math.sin((x + shift) * 0.085 / dpr) * (0.75 + 0.25 * Math.sin((x + shift) * 0.013 / dpr));
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    var hx = W / 2, ha = envAt((hx + shift) / gap);
    ctx.globalAlpha = 0.25 + 0.75 * ha; ctx.fillStyle = orange;
    ctx.beginPath(); ctx.arc(hx, mid, (2.5 + 3 * ha) * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = "destination-in";
    var g = ctx.createLinearGradient(0, 0, W, 0);
    g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(0.12, "rgba(0,0,0,1)"); g.addColorStop(0.88, "rgba(0,0,0,1)"); g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = "source-over";
  }
  function loop(now) { if (visible && !document.hidden) draw(now); requestAnimationFrame(loop); }
  window.addEventListener("resize", function () { size(); draw(performance.now()); });
  if ("IntersectionObserver" in window) new IntersectionObserver(function (e) { visible = e[0].isIntersecting; }).observe(cv);
  window.afonoSamples().then(function (s) {
    env = buildEnvelope(s); size();
    if (reduce) draw(t0 + 3000); else requestAnimationFrame(loop);
  }).catch(function () { cv.remove(); });
})();
