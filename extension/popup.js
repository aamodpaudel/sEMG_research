function send(type) {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (tabs[0]) chrome.tabs.sendMessage(tabs[0].id, { type: type }).catch(function () {});
    window.close();
  });
}
document.getElementById("hints").addEventListener("click", function () { send("toggle-hints"); });
document.getElementById("bar").addEventListener("click", function () { send("command-bar"); });
var box = document.getElementById("bridge");
chrome.storage.local.get({ bridge: true }).then(function (s) { box.checked = s.bridge; });
box.addEventListener("change", function () { chrome.storage.local.set({ bridge: box.checked }); });
fetch("http://127.0.0.1:8765/api/ping", { headers: { "X-Afono-Client": "extension" } })
  .then(function (r) { return r.ok; }).catch(function () { return false; })
  .then(function (ok) {
    document.getElementById("dot").classList.toggle("on", ok);
    document.getElementById("conn").textContent = ok ? "Connected to Nullpoint" : "Nullpoint is not running";
  });
chrome.runtime.sendMessage({ type: "status" }).then(function (s) {
  if (s) document.getElementById("count").textContent = s.lastTargets + " targets";
}).catch(function () {});
