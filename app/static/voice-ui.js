/* Afono Nullpoint, Whisper tab: microphone capture, hands-free speech detection and the self-test.
   Audio goes to the local server (/api/voice/decode) as 16 kHz int16 PCM; nothing leaves this computer. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var SR = 16000, PRE_ROLL_S = 0.3, END_SILENCE_S = 0.6, MAX_S = 5;
  var ctx = null, stream = null, node = null, rate = 48000;
  var recording = false, chunks = [], preRoll = [], busy = false;
  var floor = 0.003, loud = 0, quiet = 0, auto = false;
  var test = null, statusT = null;

  function state(t, err) { NP.msg("v-state", t, err); }

  function poll(load) {
    return NP.api("/api/voice/status" + (load ? "?load=1" : "")).then(function (s) {
      if (s.error) state("Speech model failed to load: " + s.error, true);
      else if (s.ready) state("Ready: " + s.model + " on " + s.device + (s.bundled ? ", works offline." : "."));
      else if (s.loading) state(s.bundled ? "Loading the speech model…" : "Loading the speech model (first time downloads about 290 MB)…");
      $("v-page").textContent = (s.page ? "“" + s.page + "”: " : "No page connected, Tier 1 commands only: ") +
        s.n_candidates + " phrases. " + s.candidates.slice(0, 40).join(" · ") + (s.n_candidates > 40 ? " …" : "");
      $("v-test").disabled = !(s.ready && stream);
      $("v-ptt").disabled = !(s.ready && stream);
      return s;
    }).catch(function (e) { state(e.message, true); });
  }

  function toPcm16(parts) {
    var n = 0; parts.forEach(function (p) { n += p.length; });
    var all = new Float32Array(n), o = 0;
    parts.forEach(function (p) { all.set(p, o); o += p.length; });
    var step = rate / SR, out = new Int16Array(Math.floor(n / step));
    for (var i = 0; i < out.length; i++) {                        // box-filter decimation to 16 kHz
      var a = Math.floor(i * step), b = Math.min(n, Math.floor((i + 1) * step)), s = 0;
      for (var j = a; j < b; j++) s += all[j];
      var v = s / Math.max(1, b - a);
      out[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
    }
    return out;
  }

  function startClip() { recording = true; chunks = preRoll.slice(); }
  // Speech that ends while the previous clip is still decoding is kept (the latest one) and decoded next,
  // instead of being dropped: a noise blip decoded first used to swallow the words that followed it.
  var pending = null;
  function endClip() {
    recording = false;
    var pcm = toPcm16(chunks); chunks = [];
    if (pcm.length < SR * 0.25) return;
    if (busy) { pending = pcm; return; }
    decode(pcm);
  }
  function decode(pcm) {
    busy = true;
    var q = ["send=" + (test ? 0 : ($("v-send").checked ? 1 : 0)), "gaze=" + ($("v-gaze").checked ? 1 : 0), "profile=" + encodeURIComponent(NP.prof())];
    if (test) { q.push("label=" + encodeURIComponent(test.prompts[test.i])); if ($("v-keep").checked) q.push("keep=1"); }
    NP.api("/api/voice/decode?" + q.join("&"), { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: pcm.buffer })
      .then(show).catch(function (e) { NP.msg("v-meta", e.message, true); })
      .finally(function () { busy = false; if (pending) { var p = pending; pending = null; decode(p); } });
  }

  /** Whisper transcribes silence and noise as nothing or a bracketed tag ("[BLANK_AUDIO]", "(water splashing)"). */
  function noSpeech(r) { var f = (r.free || "").trim(); return !f || /^[\[(].*[\])]$/.test(f); }

  function show(r) {
    if (noSpeech(r)) {                                   // not an attempt: never sent, never scored
      NP.msg("v-meta", "Only noise or silence in that clip (heard “" + (r.free || "") + "”), ignored." + (test ? " Say the prompt again." : ""));
      return;
    }
    $("v-out").textContent = (r.accepted ? "" : "? ") + r.say;
    NP.msg("v-meta", (r.accepted ? (r.queued ? "Sent to the page. " : "Accepted. ") : "Not sure, nothing sent. ") +
      "p = " + r.p.toFixed(2) + " of " + r.n_candidates + " phrases" + (r.gaze_used ? ", gaze used" : "") + ". Heard as “" + r.free + "”. " + r.ms + " ms.");
    $("v-top").textContent = r.top.map(function (t) { return t.p.toFixed(3) + "  " + t.say; }).join("\n");
    if (test) nextPrompt(r);
  }

  function onAudio(e) {
    var x = new Float32Array(e.inputBuffer.getChannelData(0));
    var s = 0; for (var i = 0; i < x.length; i++) s += x[i] * x[i];
    var rms = Math.sqrt(s / x.length), dt = x.length / rate;
    $("v-level").value = Math.min(1, rms * 20);
    preRoll.push(x); while (preRoll.length * dt > PRE_ROLL_S) preRoll.shift();
    if (recording) {
      chunks.push(x);
      if (auto) {
        quiet = rms < floor * 2 ? quiet + dt : 0;
        if (quiet > END_SILENCE_S) endClip();
      }
      if (chunks.length * dt > MAX_S) endClip();
      return;
    }
    if (!auto) return;
    // A clip starts after 0.2 s clearly above the room's noise floor (short clicks and bumps do not count).
    if (rms > Math.max(floor * 3.5, 0.004)) { loud += dt; if (loud > 0.2) { loud = 0; quiet = 0; startClip(); } }
    else { loud = 0; floor = 0.98 * floor + 0.02 * Math.max(rms, 0.0005); }   // slow noise-floor tracking
  }

  $("v-mic").addEventListener("click", function () {
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); node.disconnect(); ctx.close(); stream = null; $("v-mic").textContent = "Start microphone"; poll(); return; }
    // Noise suppression is off: it tends to erase whispers.
    navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true, channelCount: 1 } })
      .then(function (s) {
        stream = s; ctx = new AudioContext(); rate = ctx.sampleRate;
        var src = ctx.createMediaStreamSource(s);
        node = ctx.createScriptProcessor(2048, 1, 1);
        node.onaudioprocess = onAudio; src.connect(node); node.connect(ctx.destination);
        $("v-mic").textContent = "Stop microphone"; poll(true);
      }).catch(function (e) { state("Microphone: " + e.message, true); });
  });

  var ptt = $("v-ptt");
  function down(e) { e.preventDefault(); if (stream && !recording) startClip(); }
  function up() { if (recording && !auto) endClip(); }
  ptt.addEventListener("pointerdown", down); ptt.addEventListener("pointerup", up); ptt.addEventListener("pointerleave", up);
  $("v-auto").addEventListener("change", function (e) { auto = e.target.checked; loud = quiet = 0; });

  /* ---------- self-test ---------- */
  $("v-test").addEventListener("click", function () {
    poll().then(function (s) {
      if (!s || !s.candidates.length) return;
      var pool = s.candidates.slice(), prompts = [];
      for (var i = 0; i < 20; i++) prompts.push(pool[Math.floor(Math.random() * pool.length)]);
      test = { prompts: prompts, i: 0, rows: [], n: s.n_candidates };
      $("v-test").disabled = true;
      $("v-prompt").textContent = "Say: " + prompts[0];
      NP.msg("v-score", "1 of 20. " + (auto ? "Speak when ready." : "Hold the talk button while you speak."));
    });
  });

  function nextPrompt(r) {
    var want = test.prompts[test.i];
    test.rows.push({ ok: r.say === want, accepted: r.accepted, ms: r.ms });
    test.i += 1;
    var done = test.rows.length, ok = test.rows.filter(function (x) { return x.ok; }).length;
    if (test.i < test.prompts.length) {
      $("v-prompt").textContent = "Say: " + test.prompts[test.i];
      NP.msg("v-score", (test.i + 1) + " of 20. Correct so far " + ok + " / " + done + ".");
      return;
    }
    var acc = test.rows.filter(function (x) { return x.accepted; });
    var accOk = acc.filter(function (x) { return x.ok; }).length;
    var ms = test.rows.map(function (x) { return x.ms; }).sort(function (a, b) { return a - b; });
    var res = { accuracy: ok / done, accepted_accuracy: acc.length ? accOk / acc.length : null, reject_rate: 1 - acc.length / done,
                trials: done, median_candidates: test.n, median_ms: ms[Math.floor(ms.length / 2)], style: $("v-style").value,
                mic: $("v-micname").value.slice(0, 60), gaze: $("v-gaze").checked, profile: NP.prof() };
    $("v-prompt").textContent = "Done";
    NP.msg("v-score", "Top choice right " + ok + " / " + done + " (" + Math.round(100 * res.accuracy) + "%). " +
      "When it was sure enough to act: " + (acc.length ? Math.round(100 * accOk / acc.length) + "% right" : "never") +
      ", unsure " + Math.round(100 * res.reject_rate) + "% of the time. Logged to the run log.");
    NP.post("/api/voice/result", res).catch(function () {});
    test = null; $("v-test").disabled = false;
  }

  document.querySelector('[data-tab="voice"]').addEventListener("click", function () {
    poll(); clearInterval(statusT);
    statusT = setInterval(function () { if (!$("view-voice").hidden) poll(); }, 3000);
  });
})();
