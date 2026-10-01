/* Afono Nullpoint bridge: page targets go to the desktop app, decoded commands come back.
   Talks only to http://127.0.0.1:8765 on this computer. */
var NULLPOINT = "http://127.0.0.1:8765";
var HEADERS = { "Content-Type": "application/json", "X-Afono-Client": "extension" };
var state = { connected: false, lastTargets: 0 };

function enabled() {
  return chrome.storage.local.get({ bridge: true }).then(function (s) { return s.bridge; });
}

/* Only the tab the user is on talks to Nullpoint: the active tab of the most recently focused Chrome window.
   ("Active" alone is per window, so a second window's tab used to overwrite the page's phrases.) Focus moving to
   another app, such as the Nullpoint window, keeps the last Chrome window. */
var lastWin = null;
function currentWin() {
  if (lastWin !== null) return Promise.resolve(lastWin);
  return chrome.windows.getLastFocused({ windowTypes: ["normal"] }).then(function (w) { lastWin = w.id; return lastWin; })
    .catch(function () { return null; });
}
function refreshActive(windowId) {
  chrome.tabs.query({ active: true, windowId: windowId }, function (tabs) {
    if (tabs && tabs[0]) chrome.tabs.sendMessage(tabs[0].id, { type: "refresh" }).catch(function () {});
  });
}
chrome.windows.onFocusChanged.addListener(function (id) {
  if (id === chrome.windows.WINDOW_ID_NONE) return;
  chrome.windows.get(id, function (w) { if (w && w.type === "normal" && id !== lastWin) { lastWin = id; refreshActive(id); } });
});
chrome.tabs.onActivated.addListener(function (info) { if (info.windowId === lastWin) refreshActive(info.windowId); });
function fromCurrentTab(sender) {
  if (!sender.tab || !sender.tab.active) return Promise.resolve(false);
  return currentWin().then(function (w) { return w === null || sender.tab.windowId === w; });
}

chrome.runtime.onMessage.addListener(function (m, sender, reply) {
  var tabId = sender.tab && sender.tab.id;
  if (m.type === "targets") {
    state.lastTargets = m.targets.length;
    Promise.all([fromCurrentTab(sender), enabled()]).then(function (ok) {
      if (!ok[0] || !ok[1]) return;
      fetch(NULLPOINT + "/api/context", { method: "POST", headers: HEADERS,
        body: JSON.stringify({ url: m.url, title: m.title, targets: m.targets, vocabulary: m.vocabulary,
                               lexicon: m.lexicon, model: m.model }) })
        .then(function (r) { state.connected = r.ok; }).catch(function () { state.connected = false; });
    });
    return;
  }
  if (m.type === "poll") {
    Promise.all([fromCurrentTab(sender), enabled()]).then(function (ok) {
      if (!ok[0] || !ok[1]) { reply({ commands: [] }); return; }
      fetch(NULLPOINT + "/api/commands", { headers: HEADERS })
        .then(function (r) { state.connected = r.ok; return r.ok ? r.json() : { commands: [] }; })
        .then(function (j) { reply({ commands: j.commands || [] }); })
        .catch(function () { state.connected = false; reply({ commands: [] }); });
    });
    return true;  // async reply
  }
  if (m.type === "tab" && tabId != null) {
    if (m.action === "close") chrome.tabs.remove(tabId);
    if (m.action === "zoom") chrome.tabs.getZoom(tabId, function (z) {
      chrome.tabs.setZoom(tabId, Math.min(3, Math.max(0.3, Math.round((z + 0.1 * m.dir) * 10) / 10)));
    });
    return;
  }
  if (m.type === "status") { reply(state); return; }
  if (m.type === "ran") {
    enabled().then(function (on) {
      if (on) fetch(NULLPOINT + "/api/ran", { method: "POST", headers: HEADERS, body: JSON.stringify(m) }).catch(function () {});
    });
  }
});

chrome.commands.onCommand.addListener(function (name) {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (tabs[0]) chrome.tabs.sendMessage(tabs[0].id, { type: name }).catch(function () {});
  });
});
