/* Nullpoint Admin: live view of the volunteer window, participants (add, login, check in, delete) and results.
   Talks only to this computer's Nullpoint; every admin call carries the admin session token. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var KEY = "afono-admin-token", token = null, setup = false, people = [], live = null;
  try { token = sessionStorage.getItem(KEY); } catch (e) { /* sign in each time */ }

  function esc(t) { var d = document.createElement("div"); d.textContent = t == null ? "" : String(t); return d.innerHTML; }
  function msg(id, t, err) { var e = $(id); e.textContent = t || ""; e.classList.toggle("err", !!err); }
  function api(path, body) {
    var opt = { headers: { "X-Afono-Client": "nullpoint-admin", "X-Afono-Admin": token || "" } };
    if (body !== undefined) { opt.method = "POST"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
    return fetch(path, opt).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (r.status === 401 && /admin/i.test(j.detail || "")) { signOut(); }
        if (!r.ok) throw new Error(j.detail || ("Request failed (" + r.status + ")"));
        return j;
      });
    });
  }
  function ago(iso) {
    if (!iso) return "";
    var t = new Date(iso), s = (Date.now() - t.getTime()) / 1000;
    if (s < 90) return "just now";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " h ago";
    return t.toLocaleDateString();
  }

  /* ---------- sign in ---------- */
  function showGate(on) { $("gate").hidden = !on; if (on) setTimeout(function () { $("g-pass").focus(); }, 50); }
  function signOut() { token = null; try { sessionStorage.removeItem(KEY); } catch (e) { /* none */ } showGate(true); }
  fetch("/api/admin/state").then(function (r) { return r.json(); }).then(function (s) {
    setup = !s.set;
    if (setup) {
      $("g-title").textContent = "Set the admin password";
      $("g-lead").textContent = "First time on this laptop: choose a password for the research team (at least 8 characters). It is stored only as a salted hash here.";
      $("g-pass2-wrap").hidden = false; $("g-go").textContent = "Set password and open";
    }
    if (token) { showGate(false); refreshAll(); } else showGate(true);
  });
  $("g-form").addEventListener("submit", function (e) {
    e.preventDefault();
    if (setup && $("g-pass").value !== $("g-pass2").value) { msg("g-msg", "The two passwords are not the same.", true); return; }
    fetch(setup ? "/api/admin/setup" : "/api/admin/login", { method: "POST", headers: { "Content-Type": "application/json", "X-Afono-Client": "nullpoint-admin" },
      body: JSON.stringify({ password: $("g-pass").value }) })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || "Failed"); return j; }); })
      .then(function (j) {
        token = j.token; try { sessionStorage.setItem(KEY, token); } catch (x) { /* none */ }
        $("g-pass").value = $("g-pass2").value = ""; setup = false; showGate(false); refreshAll();
      }).catch(function (err) { msg("g-msg", err.message, true); });
  });
  $("sign-out").addEventListener("click", signOut);
  $("open-user").addEventListener("click", function () { api("/api/admin/open-user", {}).catch(function (e) { alert(e.message); }); });

  /* ---------- live ---------- */
  var MIC = { off: ["", "Microphone off"], listening: ["ok", "Listening"], hearing: ["live", "Hearing them"], decoding: ["ok", "Working it out"] };
  var STEP = { "setup": "Get ready", "st-whisper": "Voice check: whisper", "st-quiet": "Voice check: quiet voice", "run-mouse": "Task list with the mouse",
               "run-whisper": "Task list by voice", "survey": "Survey", "camera": "With the camera" };
  // Redraw a panel only when its content changes, so a click never lands on a button that was just replaced.
  function setHtml(id, html) { var e = $(id); if (e._html !== html) { e._html = html; e.innerHTML = html; } }
  $("live").addEventListener("click", function (e) {
    if (!e.target.closest("#l-out") || !live || !live.user) return;
    if (confirm("Sign " + (live.user.name || live.user.pid) + " out of the volunteer window? A task list in progress is stopped.")) api("/api/admin/signout-user", {}).then(tick);
  });
  $("run").addEventListener("click", function (e) {
    if (e.target.closest("#r-stop") && confirm("Stop this task list? Nothing from it is saved.")) api("/api/admin/stop-run", {}).then(tick);
  });

  function renderLive(l) {
    live = l;
    $("clock").textContent = l.time;
    var u = l.user, pg = l.page || {}, vo = l.voice || {}, lv = l.last_voice;
    var who = u && u.signed_in
      ? '<p class="big">' + esc(u.name || u.pid) + ' <span class="muted">(' + esc(u.pid) + ')</span></p>' +
        '<div class="kv"><span>Signed in at</span><span>' + esc(u.since) + '</span>' +
        '<span>On step</span><span>' + esc(STEP[u.step] || u.step || "–") + (u.view && u.view !== "study" ? " (" + esc(u.view) + " screen)" : "") + '</span>' +
        '<span>Microphone</span><span><span class="dot ' + (MIC[u.mic] || MIC.off)[0] + '"></span>' + esc((MIC[u.mic] || MIC.off)[1]) + '</span>' +
        '<span>Window seen</span><span>' + (u.seconds_ago <= 6 ? '<span class="dot ok"></span>now' : '<span class="dot warn"></span>' + u.seconds_ago + ' s ago (closed or asleep?)') + '</span></div>' +
        '<div class="row"><button class="btn mini" type="button" id="l-out">Sign them out</button></div>'
      : '<p class="big">No one signed in.</p><p class="muted">Check someone in from the list below, or they sign in themselves in the volunteer window.</p>';
    var sys = '<h2 style="margin-top: 6px">System</h2><div class="kv">' +
      '<span>Chrome page</span><span>' + (pg.targets ? '<span class="dot ok"></span>' + esc(pg.title || "") + ' (' + pg.targets + ' targets, ' + esc(pg.time || "") + ')' : '<span class="dot warn"></span>Not connected: open Hacker News in Chrome') + '</span>' +
      '<span>Speech model</span><span>' + (vo.ready ? '<span class="dot ok"></span>Ready on ' + esc(l.device).toUpperCase() : vo.loading ? 'Loading…' : vo.error ? '<span class="dot warn"></span>' + esc(vo.error) : 'Loads when the microphone starts') + '</span>' +
      '<span>Last heard</span><span>' + (lv ? (lv.accepted ? '<b>' + esc(lv.say) + '</b>' : '<span class="ok-q">? ' + esc(lv.say) + '</span>') +
        ' <span class="muted">heard as “' + esc(lv.heard) + '”, p ' + (lv.p || 0).toFixed(2) + ', ' + esc(lv.time) + (lv.selftest ? ", voice check" : "") + '</span>' : "–") + '</span></div>';
    setHtml("live", who + sys);

    var r = l.run;
    if (!r) { setHtml("run", '<p class="muted">No task list running.</p>'); }
    else {
      var head = '<p class="muted">' + esc(r.pid) + ', ' + (r.mode === "mouse" ? "with the mouse" : r.mode === "whisper" ? "by voice" : esc(r.mode)) + '</p>';
      var now = r.phase === "active" ? '<p class="big">Task ' + (r.i + 1) + ' of ' + r.n + ': ' + esc(r.task) + '</p><p class="timer" id="r-timer"></p>' +
          '<p class="actions">Actions so far: ' + (r.actions.length ? esc(r.actions.join(" · ")) : "none") + '</p>'
        : r.phase === "ready" ? '<p class="big">Waiting to start task ' + (r.i + 1) + ' of ' + r.n + '</p><p class="muted">' + esc(r.task) + '</p>'
        : r.phase === "finished" ? '<p class="big">Finished and saved.</p>' : '<p class="big">Saving…</p>';
      var rows = r.rows.map(function (x) {
        var m = x.result !== "done" ? '<span class="ok-n">' + esc(x.result) + '</span>' : x.verified === true ? '<span class="ok-y">✓ checked</span>' : x.verified === false ? '<span class="ok-n">ended elsewhere</span>' : '<span class="ok-q">done, not checkable</span>';
        return '<tr><td>' + x.task + '</td><td>' + esc(x.text) + '</td><td>' + x.seconds.toFixed(1) + ' s</td><td>' + m + '</td></tr>';
      }).join("");
      setHtml("run", head + now + (rows ? '<table><thead><tr><th>#</th><th>Task</th><th>Time</th><th>Result</th></tr></thead><tbody>' + rows + '</tbody></table>' : "") +
        (r.phase !== "finished" ? '<div class="row"><button class="btn mini danger" type="button" id="r-stop">Stop the task list (not saved)</button></div>' : ""));
      if ($("r-timer")) $("r-timer").textContent = (r.elapsed || 0).toFixed(1) + " s";
    }
  }
  function tick() { if (token) api("/api/admin/live").then(renderLive).catch(function () {}); }

  /* ---------- participants and results ---------- */
  function steps(p) {
    var g = p.progress, t = g.tasks || [];
    return [["W", g.selftest_whisper, "Voice check: whisper"], ["Q", g.selftest_quiet, "Voice check: quiet voice"], ["M", t.indexOf("mouse") >= 0, "Task list with the mouse"],
            ["V", t.indexOf("whisper") >= 0, "Task list by voice"], ["S", g.survey, "Survey"], ["C", g.track && t.indexOf("whisper-gaze") >= 0, "Camera"]]
      .map(function (x) { return '<i class="' + (x[1] ? "y" : "") + '" title="' + x[2] + (x[1] ? ": done" : ": not yet") + '">' + x[0] + '</i>'; }).join("");
  }
  function renderPeople(d) {
    people = d.people;
    var cur = live && live.user && live.user.signed_in ? live.user.pid : null;
    setHtml("people", people.length ? people.map(function (p) {
      return '<tr class="' + (p.pid === cur ? "current" : "") + '"><td><b>' + esc(p.pid) + '</b></td><td>' + esc(p.name) + '</td><td>' + (p.username ? esc(p.username) : '<span class="muted">none</span>') + '</td>' +
        '<td class="muted">' + esc(p.group || "") + '</td><td><div class="steps">' + steps(p) + '</div></td><td class="muted">' + esc(ago(p.last)) + '</td>' +
        '<td><div class="row" style="justify-content: flex-end"><button class="btn mini" type="button" data-act="in" data-pid="' + esc(p.pid) + '">' + (p.pid === cur ? "Checked in" : "Check in") + '</button>' +
        '<button class="btn mini" type="button" data-act="login" data-pid="' + esc(p.pid) + '">' + (p.username ? "Reset login" : "Set login") + '</button>' +
        '<button class="btn mini danger" type="button" data-act="del" data-pid="' + esc(p.pid) + '">Delete</button></div></td></tr>';
    }).join("") : '<tr><td colspan="7" class="muted">No participants yet. Volunteers create an account in the volunteer window, or add one below.</td></tr>');
    if (!$("a-group").options.length) d.groups.forEach(function (g) { var o = document.createElement("option"); o.value = g; o.textContent = g; if (g === "prefer-not-to-say") o.selected = true; $("a-group").appendChild(o); });

    var rows = d.summary.rows;
    setHtml("results", rows.length ? rows.map(function (r) {
      return '<tr><td><b>' + esc(r.pid) + '</b></td><td>' + esc(r.mode) + '</td><td>' + Math.round((r.success_rate || 0) * 100) + '%</td><td>' + (r.median_s == null ? "–" : r.median_s) + '</td><td>' + r.total_s +
        '</td><td>' + (r.verified || 0) + ' of ' + r.tasks + '</td><td>' + r.undos + '</td><td>' + (r.vs_mouse == null ? "–" : r.vs_mouse + "×") + '</td><td>' + (r.sus == null ? "–" : r.sus) + '</td></tr>';
    }).join("") : '<tr><td colspan="9" class="muted">No task lists yet.</td></tr>');
    $("ratio").textContent = Object.keys(d.summary.median_vs_mouse).map(function (k) {
      return k + ": median " + d.summary.median_vs_mouse[k].toFixed(2) + "× the same person's mouse time (1.00 = as fast as a mouse).";
    }).join(" ");
  }
  function refreshPeople() { if (token) api("/api/admin/participants").then(renderPeople).catch(function () {}); }
  function refreshAll() { tick(); refreshPeople(); }

  $("people").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-act]"); if (!b) return;
    var pid = b.dataset.pid, p = people.filter(function (x) { return x.pid === pid; })[0];
    if (b.dataset.act === "in") {
      api("/api/admin/checkin", { pid: pid }).then(function () { b.textContent = "Checking in…"; setTimeout(refreshAll, 2500); }).catch(function (err) { alert(err.message); });
    } else if (b.dataset.act === "login") {
      $("d-login-title").textContent = (p.username ? "Reset login for " : "Set login for ") + pid + " (" + p.name + ")";
      $("d-user").value = p.username || ""; $("d-pass").value = ""; msg("d-login-msg", "");
      $("d-login").dataset.pid = pid; $("d-login").showModal();
    } else if (b.dataset.act === "del") {
      $("d-del-pid").textContent = pid + " (" + p.name + ")"; $("d-confirm").value = ""; msg("d-del-msg", "");
      $("d-del").dataset.pid = pid; $("d-del").showModal();
    }
  });
  $("d-login-cancel").addEventListener("click", function () { $("d-login").close(); });
  $("d-del-cancel").addEventListener("click", function () { $("d-del").close(); });
  $("d-login-form").addEventListener("submit", function (e) {
    e.preventDefault();
    api("/api/admin/set-login", { pid: $("d-login").dataset.pid, username: $("d-user").value, password: $("d-pass").value })
      .then(function () { $("d-login").close(); refreshPeople(); }).catch(function (err) { msg("d-login-msg", err.message, true); });
  });
  $("d-del-form").addEventListener("submit", function (e) {
    e.preventDefault();
    api("/api/admin/delete", { pid: $("d-del").dataset.pid, confirm: $("d-confirm").value.trim() })
      .then(function (r) { $("d-del").close(); alert("Deleted: " + r.files + " files and " + r.registry_lines + " run-log lines."); refreshAll(); })
      .catch(function (err) { msg("d-del-msg", err.message, true); });
  });
  $("add-form").addEventListener("submit", function (e) {
    e.preventDefault();
    api("/api/admin/participants", { full_name: $("a-name").value, username: $("a-user").value.trim(), password: $("a-pass").value,
      group: $("a-group").value, consent: $("a-consent").checked, keep_voice_clips: $("a-clips").checked })
      .then(function (r) {
        msg("a-msg", "Added as " + r.pid + ".");
        ["a-name", "a-user", "a-pass"].forEach(function (id) { $(id).value = ""; }); $("a-consent").checked = $("a-clips").checked = false;
        refreshPeople();
      }).catch(function (err) { msg("a-msg", err.message, true); });
  });
  $("rebuild").addEventListener("click", function () {
    msg("rebuild-msg", "Working…");
    api("/api/admin/rebuild", {}).then(function (r) { msg("rebuild-msg", "Done: summary logged for " + r.people + " people, website numbers rebuilt. Commit when you are ready."); })
      .catch(function (err) { msg("rebuild-msg", err.message, true); });
  });

  setInterval(tick, 1000);
  setInterval(refreshPeople, 5000);
  // keep Nullpoint running while only this window is open
  setInterval(function () { fetch("/api/heartbeat", { method: "POST", headers: { "X-Afono-Client": "nullpoint-admin" } }).catch(function () {}); }, 5000);
})();
