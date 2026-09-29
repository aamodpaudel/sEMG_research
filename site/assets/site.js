/* Afono site: progress data, logo replay, hero EMG traces. */
(function () {
  "use strict";
  var P = window.AFONO_PROGRESS || null;
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function fmt(v) { return (Math.round(v * 10) / 10).toString(); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

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
    var road = document.getElementById("road");
    var label = { done: "Done", active: "In progress", next: "Next", later: "Later" };
    if (road && P.phases) {
      road.textContent = "";
      P.phases.forEach(function (p, i) {
        var li = el("li");
        li.appendChild(el("span", "ph", "Phase " + (i + 1)));
        var t = el("div"); t.appendChild(el("h3", null, p.name));
        var d = el("p", null, p.detail); d.style.cssText = "color: var(--ink-2); font-size: 15px; margin-top: 4px";
        t.appendChild(d); li.appendChild(t);
        li.appendChild(el("span", "goal", p.goal));
        var st = el("span", "st"); st.appendChild(el("span", "pill " + p.status, label[p.status] || p.status));
        li.appendChild(st); road.appendChild(li);
      });
    }
  }

  var intro = document.getElementById("intro"), replay = document.getElementById("replay");
  if (intro && replay) replay.addEventListener("click", function () {
    intro.classList.remove("intro"); void intro.offsetWidth; intro.classList.add("intro");
  });

  var copy = document.getElementById("copy-email");
  if (copy) copy.addEventListener("click", function () {
    var addr = document.querySelector(".contact .addr");
    var done = function () { copy.textContent = "Copied"; setTimeout(function () { copy.textContent = "Copy email"; }, 1600); };
    try {
      navigator.clipboard.writeText(addr.textContent.trim()).then(done, function () { selectText(addr); });
    } catch (e) { selectText(addr); }
  });
  function selectText(node) {
    var r = document.createRange(); r.selectNodeContents(node);
    var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
  }

  /* shared loader for the demo recordings */
  var FILES = window.AFONO_FILES || {};
  var samplesP = null;
  window.afonoSamples = function () {
    if (!samplesP) samplesP = Promise.all([
      fetch("demo/samples.json").then(function (r) { if (!r.ok) throw new Error("samples.json " + r.status); return r.json(); }),
      fetch(FILES.samples || "demo/samples.bin").then(function (r) { if (!r.ok) throw new Error("samples.bin " + r.status); return r.arrayBuffer(); })
    ]).then(function (x) { return { info: x[0], data: new Float32Array(x[1]) }; });
    return samplesP;
  };

  /* hero: real EMG traces drifting behind the headline */
  var cv = document.getElementById("emg-bg");
  if (!cv || !cv.getContext) return;
  var ctx = cv.getContext("2d"), trace = null, visible = true, offset = 0;
  function size() {
    var r = cv.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.max(1, Math.round(r.width * d)); cv.height = Math.max(1, Math.round(r.height * d));
  }
  function draw() {
    if (!trace) return;
    var W = cv.width, H = cv.height, C = trace.ch, n = trace.len;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--teal").trim() || "#0F5E63";
    ctx.globalAlpha = 0.28;
    ctx.lineWidth = Math.max(1, W / 1400);
    var one = cv.dataset.lines === "1";
    if (one) { C = 1; ctx.globalAlpha = 0.9; ctx.lineWidth = Math.max(1.5, W / 900); }
    var lane = one ? H / 2 : H / (C + 1), span = Math.min(n, 5000), step = span / W;
    for (var c = 0; c < C; c++) {
      ctx.beginPath();
      var y0 = one ? H / 2 : lane * (c + 1);
      for (var x = 0; x < W; x += 2) {
        var i = Math.floor(offset + x * step) % n;
        var v = trace.x[i * trace.ch + (one ? 2 : c)];
        var y = y0 + Math.max(-3, Math.min(3, v)) * lane * (one ? 0.3 : 0.16);
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }
  function tick() {
    if (visible && !document.hidden) { offset += 2.2; draw(); }
    if (!reduce) requestAnimationFrame(tick);
  }
  window.addEventListener("resize", function () { size(); draw(); });
  if ("IntersectionObserver" in window) new IntersectionObserver(function (e) { visible = e[0].isIntersecting; }).observe(cv);
  window.afonoSamples().then(function (s) {
    var m = s.info.samples[s.info.samples.length - 1];
    trace = { x: s.data.subarray(m.offset, m.offset + m.length * m.channels), len: m.length, ch: m.channels };
    size(); draw(); if (!reduce) requestAnimationFrame(tick);
  }).catch(function () { cv.remove(); });
})();
