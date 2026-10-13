/* Afono command grammar (Tier 1). Turns decoded text such as "click three", "click sign in",
   "scroll down" or "zoom in" into an action. Shared by the extension and its tests. */
(function (root) {
  "use strict";
  var NUMS = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
               eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17,
               eighteen: 18, nineteen: 19, twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70,
               eighty: 80, ninety: 90 };

  function number(words) {
    if (!words.length) return null;
    if (/^\d+$/.test(words[0])) {                                // "25", or Whisper's "2 5" -> 25
      if (words[0].length > 1) return parseInt(words[0], 10);
      var ds = words[0];
      for (var k = 1; k < 3 && /^\d$/.test(words[k] || ""); k++) ds += words[k];
      return parseInt(ds, 10);
    }
    // Digit by digit, the intent grammar's form: "two five" -> 25 (up to 3 digits). Clearer on the muscles than
    // "twenty five", where "fourteen"/"forty" differ mainly in stress.
    var digits = [];
    for (var j = 0; j < words.length && j < 3 && words[j] in NUMS && NUMS[words[j]] < 10; j++) digits.push(NUMS[words[j]]);
    if (digits.length > 1) return parseInt(digits.join(""), 10);
    var n = 0, used = false;                                     // "one hundred forty two" -> 142
    for (var i = 0; i < words.length && i < 4; i++) {
      if (words[i] === "hundred" && used) { n = n * 100; continue; }
      if (!(words[i] in NUMS)) break;
      if (used && n % 10 !== 0 && NUMS[words[i]] < 10) break;    // "twenty two three": stop after 22
      n += NUMS[words[i]]; used = true;
    }
    return used ? n : null;
  }

  function bestTarget(rest, targets) {
    var n = number(rest);
    if (n != null) return targets.filter(function (t) { return t.n === n; })[0] || null;
    var want = rest.join(" "), best = null, bestScore = 0;
    targets.forEach(function (t) {
      var lw = t.label.split(" "), overlap = 0;
      rest.forEach(function (w) { if (lw.indexOf(w) >= 0) overlap += 1; else if (lw.some(function (x) { return x.indexOf(w) === 0 && w.length > 2; })) overlap += 0.5; });
      var score = overlap / Math.max(rest.length, lw.length) + (t.label === want ? 1 : 0);
      if (score > bestScore) { bestScore = score; best = t; }
    });
    return bestScore >= 0.5 ? best : null;
  }

  /** Returns {type, ...} or {type: "unknown"}. */
  function parse(text, targets) {
    var w = (text || "").toLowerCase().replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean);
    targets = targets || [];
    if (!w.length) return { type: "unknown", text: text };
    var a = w[0], rest = w.slice(1);
    if (a === "afono") return { type: "wake" };
    if (a === "sleep") return { type: "sleep" };
    if (a === "show") return { type: "hints", on: true };
    if (a === "hide") return { type: "hints", on: false };
    if (a === "zoom") return { type: "zoom", dir: rest[0] === "out" ? -1 : 1 };
    if (a === "scroll" || a === "page") return { type: "scroll", dir: rest[0] === "up" ? -1 : 1, big: a === "page" };
    if (a === "up" || a === "down") return { type: "scroll", dir: a === "up" ? -1 : 1, big: false };
    if (a === "top" || a === "bottom") return { type: "jump", to: a };
    if (a === "back") return { type: "history", dir: -1 };
    if (a === "next" && !rest.length) return { type: "history", dir: 1 };
    if (a === "search") return { type: "search", query: rest.join(" ") };
    if (a === "close") return rest.length ? { type: "click", target: bestTarget(rest, targets) } : { type: "close" };
    if (["enter", "tab", "undo", "copy", "paste", "select", "delete", "stop", "send", "cancel", "done", "yes", "no"].indexOf(a) >= 0 && !rest.length)
      return { type: "key", key: a };
    if (a === "click" || a === "open" || a === "select") {
      var t = bestTarget(rest, targets);
      return t ? { type: "click", target: t } : { type: "unknown", text: text, reason: "no matching target" };
    }
    var direct = bestTarget(w, targets);                         // "sign in" on its own means click it
    if (direct) return { type: "click", target: direct };
    return { type: "unknown", text: text };
  }

  var api = { parse: parse, number: number };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.AfonoCommands = api;
})(typeof window !== "undefined" ? window : this);
