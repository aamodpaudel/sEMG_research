/* Afono in-browser demo: ONNX Runtime Web (WebAssembly) + lexicon CTC decoding, all on the device. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var list = $("sample-list"), scope = $("scope");
  if (!list || !scope) return;
  var D = window.AfonoDecode, S = null, sel = 0, vocab = "dates", session = null, lexicons = null, busy = false;
  var KIND = { dates: "dates", open: "sentence" };

  function status(t) { $("demo-status").textContent = t; }

  function setVocab(v) {
    vocab = v;
    $("vocab-dates").setAttribute("aria-pressed", v === "dates");
    $("vocab-open").setAttribute("aria-pressed", v === "open");
  }
  $("vocab-dates").addEventListener("click", function () { setVocab("dates"); });
  $("vocab-open").addEventListener("click", function () { setVocab("open"); });

  function drawScope(m) {
    var r = scope.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
    scope.width = Math.max(1, Math.round(r.width * d)); scope.height = Math.max(1, Math.round(r.height * d));
    var ctx = scope.getContext("2d"), W = scope.width, H = scope.height, C = m.channels, n = m.length;
    var x = S.data.subarray(m.offset, m.offset + n * C);
    ctx.clearRect(0, 0, W, H);
    var top = 30 * d, lane = (H - top - 8 * d) / C;
    for (var c = 0; c < C; c++) {
      ctx.fillStyle = c % 2 ? "rgba(240,121,74,.85)" : "rgba(127,203,196,.9)";
      var mid = top + lane * (c + 0.5);
      for (var px = 0; px < W; px++) {
        var a = Math.floor(px * n / W), b = Math.max(a + 1, Math.floor((px + 1) * n / W)), lo = 0, hi = 0;
        for (var i = a; i < b; i++) { var v = x[i * C + c]; if (v < lo) lo = v; if (v > hi) hi = v; }
        var s = lane * 0.11;
        var y1 = mid - Math.min(4.2, hi) * s, y2 = mid - Math.max(-4.2, lo) * s;
        ctx.fillRect(px, y1, 1, Math.max(1, y2 - y1));
      }
    }
    $("scope-label").textContent = C + " channels · facial EMG · 1,000 Hz · " + (n / 1000).toFixed(1) + " s · silent";
  }

  function select(i) {
    sel = i;
    var m = S.info.samples[i];
    Array.prototype.forEach.call(list.children, function (b, k) { b.setAttribute("aria-pressed", k === i); });
    setVocab(m.kind);
    $("out-ref").textContent = m.text;
    $("out-hyp").textContent = "—";
    $("out-raw").textContent = "Before the vocabulary is applied.";
    drawScope(m);
  }

  function showResult(m, hyp, raw, note) {
    var a = D.alignWords(m.text, hyp), out = $("out-hyp");
    out.textContent = "";
    if (!a.words.length) out.textContent = "(nothing recognised)";
    a.words.forEach(function (w, k) {
      if (k) out.appendChild(document.createTextNode(" "));
      var s = document.createElement("span"); s.className = w.ok ? "w-ok" : "w-bad"; s.textContent = w.word;
      out.appendChild(s);
    });
    $("out-raw").textContent = raw || "(empty)";
    var acc = Math.max(0, 1 - a.errors / Math.max(1, a.refWords));
    status("Word accuracy " + Math.round(acc * 100) + "% on this recording. " + note);
  }

  function ensureModel() {
    if (session) return Promise.resolve(session);
    if (!window.ort) return Promise.reject(new Error("runtime missing"));
    status("Loading the 9 MB model onto this device…");
    ort.env.wasm.wasmPaths = new URL("demo/ort/", document.baseURI).href;
    ort.env.wasm.numThreads = 1;
    return ort.InferenceSession.create((window.AFONO_FILES || {}).model || "demo/model.onnx", { executionProviders: ["wasm"] })
      .then(function (s) { session = s; return s; });
  }
  function ensureLexicons() {
    if (lexicons) return Promise.resolve(lexicons);
    return fetch("demo/lexicons.json").then(function (r) { return r.json(); }).then(function (j) {
      lexicons = { dates: D.buildLexicon(j.dates, S.info.chars), open: D.buildLexicon(j.open, S.info.chars) };
      return lexicons;
    });
  }

  $("decode").addEventListener("click", function () {
    if (!S || busy) return;
    busy = true; $("decode").disabled = true;
    var m = S.info.samples[sel], v = vocab;
    Promise.all([ensureModel(), ensureLexicons()]).then(function (r) {
      status("Decoding on this device…");
      var x = new Float32Array(S.data.subarray(m.offset, m.offset + m.length * m.channels));
      var t0 = performance.now();
      return r[0].run({ emg: new ort.Tensor("float32", x, [1, m.length, m.channels]) }).then(function (out) {
        var t1 = performance.now(), lp = out.logprobs.data, T = out.logprobs.dims[1], V = out.logprobs.dims[2];
        var raw = D.greedy(lp, T, V, S.info.chars);
        var hyp = D.lexiconBeam(lp, T, V, r[1][v], S.info.decoder);
        var t2 = performance.now();
        showResult(m, hyp, raw, "Ran on your device in " + Math.round(t2 - t0) + " ms (model " + Math.round(t1 - t0) +
          " ms, decoding " + Math.round(t2 - t1) + " ms) with the " + (v === "dates" ? "51" : "9,139") + "-word vocabulary.");
      });
    }).catch(function (e) {
      if (v === m.kind) {
        showResult(m, m.py_lexicon, m.py_greedy, "Your browser could not run the model here, so this shows the result our Python pipeline computed for the same recording.");
      } else {
        status("Your browser could not run the model here (" + e.message + "). Try Chrome, Edge or Firefox.");
      }
    }).then(function () { busy = false; $("decode").disabled = false; });
  });

  window.addEventListener("resize", function () { if (S) drawScope(S.info.samples[sel]); });

  window.afonoSamples().then(function (s) {
    S = s; list.textContent = "";
    s.info.samples.forEach(function (m, i) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "sample"; b.setAttribute("aria-pressed", "false");
      var t = document.createElement("span");
      t.textContent = "“" + (m.text.length > 46 ? m.text.slice(0, 44) + "…" : m.text) + "”";
      var sm = document.createElement("small");
      sm.textContent = "silent · " + KIND[m.kind] + " · " + (m.length / 1000).toFixed(1) + " s";
      b.appendChild(t); b.appendChild(sm);
      b.addEventListener("click", function () { select(i); });
      list.appendChild(b);
    });
    select(0);
  }).catch(function () {
    list.textContent = "";
    status("The demo recordings could not load. Open this page through a web server (see README), not straight from disk.");
  });
})();
