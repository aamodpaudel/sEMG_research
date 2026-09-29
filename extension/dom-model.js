/* Afono DOM Model v0: find what a person can act on, and give each target a short spoken label.
   Rule-based (accessibility semantics + pointer cursor + occlusion check). A learned detector
   for pages without semantics comes later; the fixture in test/ measures this one. */
(function (root) {
  "use strict";
  var SELECTOR = [
    "a[href]", "button", "input:not([type=hidden])", "select", "textarea", "summary",
    "[role=button]", "[role=link]", "[role=tab]", "[role=menuitem]", "[role=checkbox]", "[role=radio]",
    "[role=switch]", "[role=option]", "[role=treeitem]", "[role=combobox]", "[role=searchbox]", "[role=textbox]",
    "[onclick]", "[tabindex]:not([tabindex='-1'])", "[contenteditable='true'], [contenteditable='']"
  ].join(",");
  var NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];

  function norm(t) {
    return (t || "").toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
      .replace(/[^a-z0-9' ]+/g, " ").replace(/\s+/g, " ").trim();
  }
  function visible(el, win) {
    var r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return null;
    if (r.bottom <= 0 || r.right <= 0 || r.top >= win.innerHeight || r.left >= win.innerWidth) return null;
    var s = win.getComputedStyle(el);
    if (s.visibility === "hidden" || s.display === "none" || parseFloat(s.opacity) < 0.05) return null;
    if (el.closest("[aria-hidden='true'], [inert]")) return null;
    if (el.disabled || el.getAttribute("aria-disabled") === "true") return null;
    var cx = Math.min(win.innerWidth - 1, Math.max(0, r.left + Math.min(r.width / 2, 12)));
    var cy = Math.min(win.innerHeight - 1, Math.max(0, r.top + r.height / 2));
    var hit = el.ownerDocument.elementFromPoint(cx, cy);
    if (hit && hit !== el && !el.contains(hit) && !hit.contains(el)) {
      var lbl = hit.closest("label"); if (!(lbl && (lbl.control === el))) return null;   // covered by something else
    }
    return r;
  }
  function textOf(el) {
    var t = el.getAttribute("aria-label");
    if (!t && el.getAttribute("aria-labelledby")) {
      t = el.getAttribute("aria-labelledby").split(/\s+/).map(function (id) {
        var n = el.ownerDocument.getElementById(id); return n ? n.textContent : "";
      }).join(" ");
    }
    if (!t && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")) {
      if (el.labels && el.labels.length) t = el.labels[0].textContent;
      t = t || el.getAttribute("placeholder") || el.getAttribute("name") || (el.type === "submit" || el.type === "button" ? el.value : "");
    }
    if (!t) t = (el.innerText || el.textContent || "").split("\n").map(function (x) { return x.trim(); }).filter(Boolean)[0] || "";
    if (!t) t = el.getAttribute("title") || "";
    if (!t) { var img = el.querySelector("img[alt], svg title"); if (img) t = img.getAttribute ? (img.getAttribute("alt") || img.textContent) : ""; }
    return norm(t).split(" ").slice(0, 4).join(" ");
  }
  function kind(el) {
    var role = el.getAttribute("role"); if (role) return role;
    var tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "input") return (el.type === "checkbox" || el.type === "radio") ? el.type : (el.type === "submit" || el.type === "button" ? "button" : "field");
    if (tag === "textarea" || el.isContentEditable) return "field";
    if (tag === "select") return "menu";
    return "button";
  }

  /** Returns targets in reading order: [{n, el, label, kind, rect}]. */
  function detect(doc) {
    doc = doc || document;
    var win = doc.defaultView, seen = new Set(), found = [];
    function consider(el) {
      if (seen.has(el)) return; seen.add(el);
      var r = visible(el, win); if (!r) return;
      found.push({ el: el, rect: r });
    }
    doc.querySelectorAll(SELECTOR).forEach(consider);
    // elements that only reveal themselves through a pointer cursor (click handlers added in script)
    var all = doc.body ? doc.body.getElementsByTagName("*") : [];
    for (var i = 0; i < all.length && i < 4000; i++) {
      var el = all[i];
      if (seen.has(el) || !(el.tagName === "DIV" || el.tagName === "SPAN" || el.tagName === "LI" || el.tagName === "IMG" || el.tagName === "SVG")) continue;
      if (win.getComputedStyle(el).cursor !== "pointer") continue;
      var p = el.parentElement;
      if (p && win.getComputedStyle(p).cursor === "pointer") continue;   // keep the outermost
      consider(el);
    }
    // drop targets nested inside another target
    var set = new Set(found.map(function (f) { return f.el; }));
    found = found.filter(function (f) {
      for (var a = f.el.parentElement; a; a = a.parentElement) if (set.has(a)) return false;
      return true;
    });
    found.sort(function (a, b) { return (Math.round(a.rect.top / 8) - Math.round(b.rect.top / 8)) || (a.rect.left - b.rect.left); });
    return found.map(function (f, i) {
      var label = textOf(f.el);
      return { n: i + 1, el: f.el, label: label || kind(f.el) + " " + (i + 1), kind: kind(f.el), rect: f.rect };
    });
  }

  /** Words the sEMG Model may hear on this page: command words + target labels + numbers. */
  function vocabulary(targets) {
    var words = new Set();
    targets.forEach(function (t) { t.label.split(" ").forEach(function (w) { if (w && !/^\d+$/.test(w)) words.add(w); }); });
    NUM.forEach(function (w) { words.add(w); });
    return Array.from(words);
  }

  var api = { detect: detect, vocabulary: vocabulary, norm: norm, NUM: NUM };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.AfonoDOM = api;
})(typeof window !== "undefined" ? window : this);
