/* Afono Nullpoint page layer: number badges, a command bar, and command execution. */
(function () {
  "use strict";
  if (window.__afonoNullpoint) return;
  window.__afonoNullpoint = true;
  var D = window.AfonoDOM, C = window.AfonoCommands;
  var targets = [], hintsOn = false, asleep = false, dirty = true;

  var host = document.createElement("div");
  host.style.cssText = "all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483647;";
  var shadow = host.attachShadow({ mode: "closed" });
  shadow.innerHTML =
    "<style>" +
    ".b{position:fixed;font:600 12px/1 system-ui,sans-serif;background:#12262A;color:#F3F5F2;border:1.5px solid #F0794A;" +
    "border-radius:6px;padding:3px 5px;min-width:12px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.25);transform:translate(-4px,-4px)}" +
    ".bar{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);pointer-events:auto;display:none;align-items:center;gap:10px;" +
    "background:#12262A;border-radius:14px;padding:10px 12px;box-shadow:0 16px 40px rgba(0,0,0,.35);font:14px system-ui,sans-serif;color:#F3F5F2}" +
    ".bar.on{display:flex}.bar input{width:340px;max-width:60vw;background:#0B1B1E;color:#F3F5F2;border:1.5px solid #2E4A4E;border-radius:9px;" +
    "padding:8px 10px;font:15px system-ui,sans-serif;outline:none}.bar input:focus{border-color:#F0794A}.phi{color:#5DB5AE;font-weight:800}" +
    ".msg{position:fixed;left:50%;top:18px;transform:translateX(-50%);background:#12262A;color:#F3F5F2;border-radius:10px;padding:8px 14px;" +
    "font:14px system-ui,sans-serif;box-shadow:0 10px 30px rgba(0,0,0,.3);opacity:0;transition:opacity .2s}.msg.on{opacity:1}" +
    ".hl{position:fixed;border:2.5px solid #F0794A;border-radius:8px;transition:all .15s}" +
    "</style><div id='badges'></div><div class='hl' id='hl' hidden></div><div class='msg' id='msg'></div>" +
    "<div class='bar' id='bar'><span class='phi'>φ</span><input id='cmd' placeholder='click 3 · scroll down · zoom in · back' aria-label='Afono command'></div>";
  document.documentElement.appendChild(host);
  var badges = shadow.getElementById("badges"), bar = shadow.getElementById("bar"), cmd = shadow.getElementById("cmd");
  var msg = shadow.getElementById("msg"), hl = shadow.getElementById("hl"), msgT = null;

  function toast(t) { msg.textContent = t; msg.classList.add("on"); clearTimeout(msgT); msgT = setTimeout(function () { msg.classList.remove("on"); }, 1600); }
  function refresh() {
    targets = D.detect(document); dirty = false;
    if (hintsOn) drawBadges();
    // "vocabulary" stays a flat word list (Nullpoint's page lexicon); "lexicon" splits it for newer readers.
    var lex = D.lexicon(targets.slice(0, 150), { doc: document });
    chrome.runtime.sendMessage({ type: "targets", url: location.href, title: document.title,
      targets: targets.slice(0, 150).map(function (t) { return { n: t.n, label: t.label, kind: t.kind, name: (t.name || "").slice(0, 120), rect: screenRect(t.rect) }; }),
      vocabulary: lex.all, lexicon: { labels: lex.labels, commands: lex.commands, numbers: lex.numbers },
      model: D.version }).catch(function () {});
  }
  // Screen CSS px, as the Track Model expects (same window-frame estimate as app/static/track-ui.js).
  function screenRect(r) {
    var fs = !!document.fullscreenElement, border = fs ? 0 : Math.max(0, (outerWidth - innerWidth) / 2);
    var x = fs ? 0 : screenX + border, y = fs ? 0 : screenY + Math.max(0, outerHeight - innerHeight - border);
    return { left: Math.round(r.left + x), top: Math.round(r.top + y), width: Math.round(r.width), height: Math.round(r.height) };
  }
  function drawBadges() {
    badges.textContent = "";
    targets.slice(0, 150).forEach(function (t) {
      var b = document.createElement("div"); b.className = "b"; b.textContent = t.n;
      b.style.left = Math.max(2, t.rect.left) + "px"; b.style.top = Math.max(2, t.rect.top) + "px";
      badges.appendChild(b);
    });
  }
  function setHints(on) { hintsOn = on; if (on) { refresh(); } else badges.textContent = ""; }
  function flash(t) {
    var r = (t.click || t.el).getBoundingClientRect();
    hl.hidden = false; hl.style.left = r.left - 3 + "px"; hl.style.top = r.top - 3 + "px"; hl.style.width = r.width + 6 + "px"; hl.style.height = r.height + 6 + "px";
    setTimeout(function () { hl.hidden = true; }, 500);
  }
  function key(name) {
    var el = document.activeElement || document.body, map = { enter: "Enter", tab: "Tab", "delete": "Backspace" };
    if (name === "copy" || name === "paste" || name === "undo" || name === "delete") { try { document.execCommand(name); } catch (e) {} return; }
    if (name === "select") { try { document.execCommand("selectAll"); } catch (e) {} return; }
    if (name === "stop") { window.stop(); return; }
    if (name === "tab") {
      var i = targets.findIndex(function (t) { return t.el === el; }), nx = targets[(i + 1) % Math.max(1, targets.length)];
      if (nx) { nx.el.focus(); flash(nx); } return;
    }
    if (name === "enter" || name === "send" || name === "yes" || name === "done") {
      if (el.form && el.form.requestSubmit) { el.form.requestSubmit(); return; }
      el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })); if (el.click && el !== document.body) el.click(); return;
    }
    if (name === "cancel" || name === "no") { document.activeElement && document.activeElement.blur(); return; }
    if (map[name]) el.dispatchEvent(new KeyboardEvent("keydown", { key: map[name], bubbles: true }));
  }
  function run(text, from) {
    if (dirty) refresh();
    var a = C.parse(text, targets);
    if (asleep && a.type !== "wake") return;
    switch (a.type) {
      case "wake": asleep = false; toast("φ listening"); break;
      case "sleep": asleep = true; toast("φ asleep. Say afono to wake."); break;
      case "hints": setHints(a.on); break;
      case "scroll": window.scrollBy({ top: a.dir * (a.big ? window.innerHeight * 0.9 : 320), behavior: "smooth" }); break;
      case "jump": window.scrollTo({ top: a.to === "top" ? 0 : document.documentElement.scrollHeight, behavior: "smooth" }); break;
      case "history": history.go(a.dir); break;
      case "zoom": case "close": chrome.runtime.sendMessage({ type: "tab", action: a.type, dir: a.dir }).catch(function () {}); break;
      case "key": key(a.key); break;
      case "search":
        var box = targets.filter(function (t) { return t.kind === "field" || t.kind === "searchbox" || t.kind === "combobox"; })[0];
        if (box) { box.el.focus(); box.el.value = a.query; box.el.dispatchEvent(new Event("input", { bubbles: true })); if (box.el.form && box.el.form.requestSubmit) box.el.form.requestSubmit(); }
        else toast("No search box on this page"); break;
      case "click":
        flash(a.target); toast("click " + a.target.n + " · " + (a.target.display || a.target.label));
        setTimeout(function () {
          var k = a.target.kind;
          if (k === "field" || k === "searchbox" || k === "combobox") a.target.el.focus(); else (a.target.click || a.target.el).click();
          dirty = true;
        }, 180); break;
      default: toast("Did not understand “" + text + "”");
    }
    chrome.runtime.sendMessage({ type: "ran", text: text, action: a.type, from: from }).catch(function () {});
  }

  cmd.addEventListener("keydown", function (e) {
    e.stopPropagation();
    if (e.key === "Enter") { var t = cmd.value; cmd.value = ""; run(t, "typed"); }
    if (e.key === "Escape") { bar.classList.remove("on"); cmd.blur(); }
  });
  chrome.runtime.onMessage.addListener(function (m) {
    if (m.type === "toggle-hints") setHints(!hintsOn);
    if (m.type === "command-bar") { bar.classList.add("on"); setHints(true); setTimeout(function () { cmd.focus(); }, 30); }
    if (m.type === "run") run(m.text, "nullpoint");
  });
  var mo = new MutationObserver(function () { dirty = true; });
  mo.observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ["class", "style", "hidden", "disabled"] });
  window.addEventListener("scroll", function () { dirty = true; if (hintsOn) requestAnimationFrame(refresh); }, { passive: true });
  window.addEventListener("resize", function () { dirty = true; });
  // keep Nullpoint's vocabulary current, and pick up commands it decoded
  setInterval(function () {
    if (document.hidden) return;
    if (dirty) refresh();
    chrome.runtime.sendMessage({ type: "poll" }).then(function (r) { (r && r.commands || []).forEach(function (c) { run(c, "nullpoint"); }); }).catch(function () {});
  }, 700);
  refresh();
})();
