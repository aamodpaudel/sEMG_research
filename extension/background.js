/* Afono Nullpoint bridge: page targets go to the desktop app, decoded commands come back.
   Talks only to http://127.0.0.1:8765 on this computer. */
var NULLPOINT = "http://127.0.0.1:8765";
var HEADERS = { "Content-Type": "application/json", "X-Afono-Client": "extension" };
var state = { connected: false, lastTargets: 0 };

function enabled() {
  return chrome.storage.local.get({ bridge: true }).then(function (s) { return s.bridge; });
}

chrome.runtime.onMessage.addListener(function (m, sender, reply) {
  var tabId = sender.tab && sender.tab.id;
  if (m.type === "targets") {
    state.lastTargets = m.targets.length;
    if (!sender.tab || !sender.tab.active) return;
    enabled().then(function (on) {
      if (!on) return;
      fetch(NULLPOINT + "/api/context", { method: "POST", headers: HEADERS,
        body: JSON.stringify({ url: m.url, title: m.title, targets: m.targets, vocabulary: m.vocabulary }) })
        .then(function (r) { state.connected = r.ok; }).catch(function () { state.connected = false; });
    });
    return;
  }
  if (m.type === "poll") {
    if (!sender.tab || !sender.tab.active) { reply({ commands: [] }); return; }
    enabled().then(function (on) {
      if (!on) { reply({ commands: [] }); return; }
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
