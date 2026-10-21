/* Afono Nullpoint, Study tab: sign-in, the guided session, the task list runner and the survey.
   Accounts and answers stay on this computer (data/study); the run log only gets the pseudonym and numbers. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var KEY = "afono-study-token";
  var token = null;
  try { token = sessionStorage.getItem(KEY); } catch (e) { /* private window: sign in each time */ }
  var me = null, current = null, run = null, runT = null, meT = null, spokenFor = null;
  // "/" is the volunteer app (only the study); "/lab" keeps every research tab.
  var LAB = location.pathname.indexOf("/lab") === 0;
  if (!LAB) document.body.classList.add("user-app");

  function view() { var v = document.querySelector('[role="tab"][aria-selected="true"]'); return v ? v.dataset.tab : "study"; }
  function goTo(tab, label) {
    $("tab-" + tab).click();
    document.body.classList.toggle("away", tab !== "study");
    if (label) $("back-text").textContent = label;
  }
  $("back-btn").addEventListener("click", function () { goTo("study"); load(); });

  function api(path, body) {
    var opt = { headers: { "X-Afono-Token": token || "" } };
    if (body !== undefined) { opt.method = "POST"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
    return NP.api(path, opt);
  }
  function esc(t) { var d = document.createElement("div"); d.textContent = t == null ? "" : String(t); return d.innerHTML; }
  function device() { try { return localStorage.getItem("afono-device-" + (me ? me.pid : "")) || ""; } catch (e) { return ""; } }
  function setDevice(v) { try { localStorage.setItem("afono-device-" + me.pid, v); } catch (e) { /* not kept */ } }

  /* ---------- sign in ---------- */
  function showGate(on) { $("gate").hidden = !on; if (on) setTimeout(function () { ($("f-new").hidden ? $("l-user") : $("r-name")).focus(); }, 50); }
  function pick(isNew) {
    $("g-new").setAttribute("aria-pressed", isNew); $("g-back").setAttribute("aria-pressed", !isNew);
    $("f-new").hidden = !isNew; $("f-back").hidden = isNew;
  }
  $("g-new").addEventListener("click", function () { pick(true); });
  $("g-back").addEventListener("click", function () { pick(false); });
  $("g-skip").addEventListener("click", function () { showGate(false); render(); });
  $("f-new").addEventListener("submit", function (e) {
    e.preventDefault();
    if (!$("r-consent").checked) { NP.msg("r-msg", "Please tick that you agree to take part.", true); return; }
    NP.post("/api/study/register", { full_name: $("r-name").value, username: $("r-user").value, password: $("r-pass").value,
      password2: $("r-pass2").value, consent: true, keep_voice_clips: $("r-clips").checked, group: $("r-group").value })
      .then(signedIn).catch(function (err) { NP.msg("r-msg", err.message, true); });
  });
  $("f-back").addEventListener("submit", function (e) {
    e.preventDefault();
    NP.post("/api/study/login", { username: $("l-user").value, password: $("l-pass").value })
      .then(signedIn).catch(function (err) { NP.msg("l-msg", err.message, true); });
  });
  function signedIn(r) {
    token = r.token;
    try { sessionStorage.setItem(KEY, token); } catch (e) { /* not kept */ }
    ["r-pass", "r-pass2", "l-pass"].forEach(function (id) { $(id).value = ""; });
    showGate(false); current = null; load();
  }
  $("sign-out").addEventListener("click", function () {
    api("/api/study/logout", {}).catch(function () {}).finally(signedOut);
  });

  function signedOut() {
    token = null; me = null; run = null; current = null; stopRun();
    try { sessionStorage.removeItem(KEY); } catch (e) { /* nothing kept */ }
    var p = $("profile"); p.readOnly = false; p.value = "default"; p.dispatchEvent(new Event("change"));
    document.body.classList.remove("signed-in"); $("acct").hidden = true;
    goTo("study"); render(); pick(false); if (!LAB) showGate(true);
  }
  // The admin app can check a participant in: the volunteer window picks the session up here.
  setInterval(function () {
    if (token || LAB || $("gate").hidden) return;
    NP.api("/api/study/checkin").then(function (r) { if (r && r.token) signedIn(r); }).catch(function () {});
  }, 2000);
  // Tell the admin app where this window is; a 401 means the admin signed this person out.
  setInterval(function () {
    if (!token) return;
    api("/api/study/ping", { step: current, view: view(), mic: NP.micState ? NP.micState() : "" })
      .catch(function (e) { if (/sign in/i.test(e.message)) signedOut(); });
  }, 3000);

  function load() {
    if (!token) { render(); if (!LAB) showGate(true); return Promise.resolve(); }
    return api("/api/study/me").then(function (m) {
      var first = !me;
      me = m;
      var p = $("profile");
      if (p.value !== m.pid) { p.value = m.pid; p.dispatchEvent(new Event("change")); }
      p.readOnly = true;
      document.body.classList.add("signed-in");
      $("acct").hidden = false; $("acct-name").textContent = m.first; $("acct-pid").textContent = m.pid; $("st-pid").textContent = m.pid;
      if (m.run) { run = m.run; if (run.phase !== "finished") startRun(); }
      if (first || !current) current = firstOpen();
      render();
    }).catch(function () { signedOut(); });
  }

  /* ---------- the steps ---------- */
  function pageOk() { return !!(me && me.page && me.page.targets > 0 && /hacker news/i.test(me.page.title || "")); }
  function steps() {
    if (!me) return [];
    var p = me.progress, first = p.first_mode, second = first === "mouse" ? "whisper" : "mouse";
    function runStep(mode) {
      return { id: "run-" + mode, mode: mode, title: mode === "mouse" ? "Task list with the mouse" : "Task list by voice",
               sub: "7 short tasks on Hacker News", done: p.tasks.indexOf(mode) >= 0 };
    }
    return [
      { id: "setup", title: "Get ready", sub: "Chrome, earbuds, microphone", done: pageOk() && NP.micOn() },
      { id: "st-whisper", style: "whisper", title: "Voice check: whisper", sub: "20 short phrases", done: p.selftest_whisper },
      { id: "st-quiet", style: "quiet", title: "Voice check: quiet voice", sub: "20 short phrases", done: p.selftest_quiet },
      runStep(first), runStep(second),
      { id: "survey", title: "Your opinion", sub: "10 quick questions and 4 open ones", done: p.survey },
      { id: "camera", mode: "whisper-gaze", later: true, title: "With the camera", sub: "Next session: look, then say “click”",
        done: p.track && p.tasks.indexOf("whisper-gaze") >= 0 }
    ];
  }
  function firstOpen() { var s = steps().filter(function (x) { return !x.done && !x.later; })[0]; return s ? s.id : "survey"; }
  function step(id) { return steps().filter(function (x) { return x.id === id; })[0]; }
  function next() {
    var all = steps(), i = all.map(function (x) { return x.id; }).indexOf(current);
    var after = all.slice(i + 1).filter(function (x) { return !x.done; })[0];
    current = after ? after.id : current; render();
  }

  function render() { renderSteps(); renderChecks(); renderMain(); }
  function renderSteps() {
    var box = $("st-steps"); box.innerHTML = "";
    steps().forEach(function (s, i) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "step" + (s.done ? " done" : "") + (s.later ? " later" : "");
      b.setAttribute("aria-current", s.id === current);
      b.innerHTML = '<span class="n">' + (s.done ? "✓" : i + 1) + '</span><span>' + esc(s.title) + '<small>' + esc(s.sub) + '</small></span>';
      b.addEventListener("click", function () { current = s.id; render(); });
      box.appendChild(b);
    });
  }
  function checkRow(ok, text) { return '<div class="check-row"><span class="' + (ok ? "ok-y" : "ok-n") + '">' + (ok ? "✓" : "✕") + '</span><span>' + text + '</span></div>'; }
  function renderChecks() {
    var box = $("st-checks");
    if (!me) { box.innerHTML = '<p>Sign in to see your checks.</p>'; return; }
    var pg = me.page || {};
    box.innerHTML = checkRow(pageOk(), pageOk() ? "Chrome is on Hacker News (" + pg.targets + " things you can click)" : "Open https://news.ycombinator.com in Chrome") +
      checkRow(NP.micOn(), NP.micOn() ? "Microphone is on" : "Microphone is off") +
      checkRow(!!$("v-auto").checked, $("v-auto").checked ? "Listening hands-free" : "Hands-free listening is off");
  }

  /* ---------- main panel, per step ---------- */
  function renderMain() {
    var box = $("st-main");
    if (!me) {
      box.innerHTML = '<h2>Study</h2><div class="panel"><p class="big-task">' + (LAB ? "Research tools: sign in only to test the study." : "Sign in to take part.") + '</p>' +
        '<p>Each volunteer has their own account, so their results stay separate.</p><div class="row"><button class="btn primary" type="button" id="st-signin">Sign in or create an account</button></div></div>';
      $("st-signin").addEventListener("click", function () { showGate(true); });
      return;
    }
    var s = step(current) || step(firstOpen());
    current = s.id;
    if (s.id === "setup") return renderSetup(box);
    if (s.style) return renderSelftest(box, s);
    if (s.mode) return renderRunStep(box, s);
    if (s.id === "survey") return renderSurvey(box);
  }

  function micOnHandsFree() {
    if (!$("v-auto").checked) { $("v-auto").checked = true; $("v-auto").dispatchEvent(new Event("change")); }
    if (!NP.micOn()) $("v-mic").click();
    setTimeout(renderChecks, 1500);
  }

  function renderSetup(box) {
    box.innerHTML = '<h2>Step 1. Get ready</h2><div class="panel">' +
      '<p class="big-task">Hi ' + esc(me.first) + '. Three things before we start.</p>' +
      '<ol class="steps" style="font-size: 15px; gap: 12px">' +
      '<li><b>Chrome on the left, this window on the right.</b> In Chrome, open <code>news.ycombinator.com</code>. Keep both windows visible the whole time.</li>' +
      '<li><b>Check the connection.</b> Click the φ icon at the top right of Chrome: it should say <b>Connected to Nullpoint</b>. The box on the right of this window turns green.</li>' +
      '<li><b>Your earbuds.</b> Put them in and choose them as the microphone in Windows (Settings, System, Sound, Input). Which ones are they?' +
      '<input type="text" id="st-device" placeholder="e.g. AirPods Pro, JLab Go Air, laptop microphone" style="width: 100%; margin-top: 8px"></li>' +
      '<li><b>Turn on the microphone.</b> The bar at the top of this window moves when you speak.<div class="row" style="margin-top: 8px"><button class="btn primary" type="button" id="st-mic">Turn on the microphone</button></div></li>' +
      '</ol><p>Try it: say <span class="say">scroll down</span> in a quiet voice. Hacker News should scroll.</p>' +
      '<div class="row"><button class="btn primary" type="button" id="st-next">Everything works, next step</button></div><p class="msg" id="st-msg"></p></div>';
    $("st-device").value = device();
    $("st-device").addEventListener("input", function (e) { setDevice(e.target.value); });
    $("st-mic").addEventListener("click", micOnHandsFree);
    $("st-next").addEventListener("click", function () {
      if (!device()) { NP.msg("st-msg", "Please write which earbuds or microphone you use.", true); return; }
      if (!NP.micOn()) { NP.msg("st-msg", "Turn on the microphone first.", true); return; }
      if (!pageOk()) { NP.msg("st-msg", "Nullpoint cannot see Hacker News in Chrome yet. Open it, then click the φ icon to check.", true); return; }
      next();
    });
  }

  function renderSelftest(box, s) {
    var whisper = s.style === "whisper";
    box.innerHTML = '<h2>' + esc(s.title) + '</h2><div class="panel">' +
      '<p class="big-task">' + (whisper ? "Whisper: breath only, no voice at all." : "Quiet voice: your real voice, very soft.") + '</p>' +
      '<p>' + (whisper ? "Like whispering to someone during a film. If it does not hear you, whisper a little louder, but do not use your voice."
                       : "Like talking to the person next to you in a library.") + '</p>' +
      '<p>Nullpoint shows 20 short phrases, one at a time, in big letters. Say each one; the next appears by itself. It takes about 2 minutes. Nothing happens in Chrome during the check.</p>' +
      (s.done ? '<p><b class="ok-y">✓ Done.</b> You can do it again if you like; the newest one counts.</p>' : "") +
      '<div class="row"><button class="btn primary" type="button" id="st-go">Start the voice check</button></div><p class="msg" id="st-msg"></p></div>';
    $("st-go").addEventListener("click", function () {
      $("v-style").value = s.style; $("v-micname").value = device() || "not given"; $("v-keep").checked = !!me.keep_voice_clips;
      micOnHandsFree();
      goTo("voice", "Voice check: say each phrase shown under Self-test, on the right. It takes about 2 minutes.");
      var tries = 0;
      (function wait() {          // the speech model may still be loading
        if (!$("v-test").disabled) { $("v-test").click(); $("v-prompt").scrollIntoView({ block: "center" }); return; }
        if (++tries < 60) setTimeout(wait, 500);
      })();
    });
  }

  var SAY = ["click two five", "show", "hide", "scroll down", "scroll up", "top", "bottom", "back", "zoom in", "zoom out", "new", "guidelines"];
  function renderRunStep(box, s) {
    var voice = s.mode !== "mouse", mine = run && run.mode === s.mode;
    if (mine && run.phase !== "finished") return renderRun(box, s);
    var intro = voice
      ? '<p class="big-task">Hands in your lap: you only speak.</p>' +
        '<p>Say <span class="say">show</span> to put a number on everything you can click, then <span class="say">click</span> and the number, digit by digit: <span class="say">click two five</span> for 25. Or say a link’s words, like <span class="say">new</span>.</p>' +
        '<div class="say-list">' + SAY.map(function (w) { return '<span class="say">' + w + '</span>'; }).join("") + '</div>' +
        '<p>Each task: say <span class="say">start</span>, do it, then say <span class="say">done</span>. Say <span class="say">skip</span> if you cannot do it.</p>' +
        (s.mode === "whisper-gaze" ? '<p>With the camera: look at a link and just say <span class="say">click</span>. First set up the camera: click <b>Start camera</b>, then <b>Calibrate</b> and look at each dot, then <b>Accuracy test</b>, <b>Target test</b> and <b>Live snap</b>.</p>' +
          '<div class="row"><button class="btn" type="button" id="st-cam">Set up the camera</button></div>' : "") +
        '<label class="f" for="st-style">How you will speak<select id="st-style"><option value="quiet">Quiet voice</option><option value="whisper">Whisper</option></select></label>'
      : '<p class="big-task">Use the mouse and keyboard as you normally would.</p>' +
        '<p>For each task: click <b>Start</b> here, do it in Chrome, then click <b>Done</b> here. Zoom with Ctrl and the mouse wheel.</p>';
    var result = mine && run.phase === "finished" ? resultsHtml(run) : (s.done ? '<p><b class="ok-y">✓ Done.</b> Running it again adds a new run; the newest one counts.</p>' : "");
    box.innerHTML = '<h2>' + esc(s.title) + '</h2><div class="panel">' + intro +
      '<p>Before you start, Chrome should show the top of Hacker News, zoom at 100% (Ctrl+0).</p>' +
      '<label class="check"><input type="checkbox" id="st-read"> Read each task aloud to me</label>' +
      '<div class="row"><button class="btn primary" type="button" id="st-start">' + (mine ? "Run it again" : "Start the task list") + '</button>' +
      (mine && run.phase === "finished" ? '<button class="btn" type="button" id="st-on">Next step</button>' : "") + '</div><p class="msg" id="st-msg"></p>' + result + '</div>';
    try { $("st-read").checked = localStorage.getItem("afono-read") === "1"; } catch (e) { /* default off */ }
    $("st-read").addEventListener("change", function (e) { try { localStorage.setItem("afono-read", e.target.checked ? "1" : "0"); } catch (x) { /* not kept */ } });
    if ($("st-on")) $("st-on").addEventListener("click", next);
    if ($("st-cam")) $("st-cam").addEventListener("click", function () {
      $("v-gaze").checked = true;
      goTo("track", "Camera: Start camera, Calibrate (look at each dot), Accuracy test, Target test, then Live snap. Then come back.");
    });
    $("st-start").addEventListener("click", function () {
      if (voice && !NP.micOn()) { NP.msg("st-msg", "Turn on the microphone first (step 1).", true); return; }
      var style = voice ? $("st-style").value : "";
      api("/api/study/run/start", { mode: s.mode, device: device(), style: style }).then(function (v) { run = v; spokenFor = null; startRun(); render(); })
        .catch(function (err) { NP.msg("st-msg", err.message, true); });
    });
  }

  function resultsHtml(v) {
    var rows = v.rows.map(function (r) {
      var mark = r.result !== "done" ? '<span class="ok-n">✕</span>' : r.verified === true ? '<span class="ok-y">✓</span>' : r.verified === false ? '<span class="ok-n">?</span>' : '<span class="ok-q">✓</span>';
      return '<div>' + mark + '<span>' + esc(r.text) + '</span><span class="msg">' + esc(r.result === "done" ? "" : r.result) + '</span><b>' + r.seconds.toFixed(1) + ' s</b></div>';
    }).join("");
    var m = v.metrics || {};
    return '<h2 style="margin-top: 8px">Your run</h2><div class="rows">' + rows + '</div>' +
      (v.metrics ? '<p>Saved. ' + Math.round((m.success_rate || 0) * 100) + '% of tasks done, median ' + (m.median_s == null ? "–" : m.median_s + " s") + ' per task.</p>' : "");
  }

  function renderRun(box, s) {
    var voice = s.mode !== "mouse", v = run, t = v.task || "";
    var body;
    if (v.phase === "ready") {
      body = '<p class="tag">Task ' + (v.i + 1) + ' of ' + v.n + '</p><p class="big-task">' + esc(t) + '</p>' +
        '<p>' + (voice ? 'Say <span class="say">start</span> when you are ready.' : 'Click <b>Start</b>, then do it in Chrome.') + '</p>' +
        '<div class="row"><button class="btn primary" type="button" id="st-begin">Start</button></div>';
    } else if (v.phase === "active") {
      body = '<p class="tag">Task ' + (v.i + 1) + ' of ' + v.n + '</p><p class="big-task">' + esc(t) + '</p>' +
        '<p class="timer" id="st-timer">' + (v.elapsed || 0).toFixed(1) + ' s</p>' +
        '<p>' + (voice ? 'Say <span class="say">done</span> when it is done, or <span class="say">skip</span>.' : 'Click <b>Done</b> when it is done.') + '</p>' +
        '<div class="row"><button class="btn primary" type="button" id="st-done">Done</button><button class="btn" type="button" id="st-skip">Skip</button>' +
        '<button class="btn" type="button" id="st-fail">I give up</button></div>';
    } else {
      body = '<p class="big-task">Saving your run…</p>';
    }
    box.innerHTML = '<h2>' + esc(s.title) + '</h2><div class="panel">' + body +
      (v.rows.length ? resultsHtml(v) : "") + '<button class="linkbtn" type="button" id="st-cancel" style="justify-self: start">Stop the task list (nothing is saved)</button></div>';
    if ($("st-begin")) $("st-begin").addEventListener("click", function () { api("/api/study/run/begin", {}).then(update); });
    if ($("st-done")) $("st-done").addEventListener("click", function () { api("/api/study/run/finish", { result: "done" }).then(update); });
    if ($("st-skip")) $("st-skip").addEventListener("click", function () { api("/api/study/run/finish", { result: "skipped" }).then(update); });
    if ($("st-fail")) $("st-fail").addEventListener("click", function () { api("/api/study/run/finish", { result: "failed" }).then(update); });
    $("st-cancel").addEventListener("click", function () { api("/api/study/run/cancel", {}).then(function () { run = null; stopRun(); render(); }); });
    if (v.phase === "ready" && spokenFor !== v.i) speak(v);
  }

  function speak(v) {
    spokenFor = v.i;
    var on = false;
    try { on = localStorage.getItem("afono-read") === "1"; } catch (e) { /* off */ }
    if (!on || !window.speechSynthesis) return;
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance("Task " + (v.i + 1) + ". " + v.task));
  }

  /* ---------- live run state (voice commands change it on the server) ---------- */
  function update(v) {
    var before = run ? run.phase + run.i : "";
    run = v;
    if (!v) { stopRun(); render(); return; }
    if (v.phase === "finished") { stopRun(); load(); return; }
    var s = step(current);
    if (!s || s.mode !== v.mode) { current = "run-" + v.mode; if (!step(current)) current = "camera"; }
    if (before !== v.phase + v.i) render();
    else if (v.phase === "active" && $("st-timer")) $("st-timer").textContent = (v.elapsed || 0).toFixed(1) + " s";
  }
  function startRun() {
    if (!runT) runT = setInterval(function () {
      api("/api/study/run").then(update).catch(function (e) { if (/sign in/i.test(e.message)) signedOut(); });
    }, 400);
  }
  function stopRun() { clearInterval(runT); runT = null; }
  document.addEventListener("np-study", function () { api("/api/study/run").then(update).catch(function () {}); });

  /* ---------- survey ---------- */
  function renderSurvey(box) {
    var scale = function (name) {
      return '<div class="scale"><span>Strongly disagree</span>' + [1, 2, 3, 4, 5].map(function (n) {
        return '<label><input type="radio" name="' + name + '" value="' + n + '"><span>' + n + '</span></label>';
      }).join("") + '<span>Strongly agree</span></div>';
    };
    box.innerHTML = '<h2>Your opinion</h2><div class="panel" style="gap: 4px">' +
      '<p>About <b>using Nullpoint by voice</b>. There are no right answers; honest ones help most.</p>' +
      me.sus.map(function (q, i) { return '<div class="sus-q"><span>' + (i + 1) + '. ' + esc(q) + '</span>' + scale("q" + i) + '</div>'; }).join("") +
      '<div class="sus-q"><span>If using a mouse or keyboard hurt, how likely are you to use this?</span><div class="scale"><span>Not at all</span>' +
      [1, 2, 3, 4, 5].map(function (n) { return '<label><input type="radio" name="would" value="' + n + '"><span>' + n + '</span></label>'; }).join("") + '<span>Very likely</span></div></div>' +
      me.open.map(function (o) { return '<div class="sus-q"><label class="f" for="o-' + o[0] + '" style="text-transform: none; letter-spacing: 0; font: 500 15px var(--body); color: var(--ink)">' + esc(o[1]) +
        '<textarea id="o-' + o[0] + '" rows="2"></textarea></label></div>'; }).join("") +
      '<div class="row" style="margin-top: 10px"><button class="btn primary" type="button" id="st-send">Send my answers</button></div><p class="msg" id="st-msg"></p></div>';
    $("st-send").addEventListener("click", function () {
      var pickV = function (name) { var x = document.querySelector('input[name="' + name + '"]:checked'); return x ? +x.value : 0; };
      var answers = me.sus.map(function (_, i) { return pickV("q" + i); });
      var missing = answers.map(function (a, i) { return a ? null : i + 1; }).filter(Boolean);
      if (missing.length || !pickV("would")) { NP.msg("st-msg", "Please answer every question" + (missing.length ? " (missing: " + missing.join(", ") + ")" : "") + ".", true); return; }
      var open = {}; me.open.forEach(function (o) { open[o[0]] = $("o-" + o[0]).value; });
      api("/api/study/survey", { answers: answers, would_use: pickV("would"), open: open }).then(function (r) {
        box.innerHTML = '<h2>Your opinion</h2><div class="panel"><p class="big-task">Thank you, ' + esc(me.first) + '.</p>' +
          '<p>Your answers are saved. Usability score ' + r.sus.toFixed(1) + ' out of 100 (68 is average).</p>' +
          '<p>If you know someone with RSI, arthritis or limited hand use who might like to try Nullpoint, please tell them about it.</p>' +
          '<div class="row"><button class="btn" type="button" id="st-out">Sign out</button></div></div>';
        $("st-out").addEventListener("click", function () { $("sign-out").click(); });
        load();
      }).catch(function (err) { NP.msg("st-msg", err.message, true); });
    });
  }

  /* ---------- keep the checks and progress current ---------- */
  document.addEventListener("np-selftest-done", function () { setTimeout(function () { goTo("study"); load().then(next); }, 1800); });
  meT = setInterval(function () {
    if (!me || runT || $("view-study").hidden) return;
    api("/api/study/me").then(function (m) {
      var was = JSON.stringify([me.progress, pageOk()]);
      me = m;
      renderChecks();
      if (JSON.stringify([me.progress, pageOk()]) !== was) renderSteps();
    }).catch(function () {});
  }, 3000);
  load();
})();
