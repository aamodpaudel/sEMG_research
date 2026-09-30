/* Afono DOM Model v1: find what a person can act on, name it the way a screen reader would, and give each
   target a short spoken label (1-2 words, numbered when duplicated) plus a numeric badge.
   Rule-based, no machine learning:
     1. candidates: native controls, ARIA widget roles, click handlers, and pointer-cursor elements,
        including inside open shadow roots (iframes are out of scope);
     2. filters: rendered, not disabled, not aria-hidden/inert, not covered by another element;
     3. accessible name: W3C accname 1.2 (aria-labelledby, aria-label, <label>, alt, title, placeholder, content);
     4. short label: drop stop words and counts, keep the first content words, spell out small numbers.
   The label words become the decoder's lexicon for this page (vocabulary()). A learned detector for pages
   built from unlabeled clickable <div>s comes later (see extension/test/webui/). */
(function (root) {
  "use strict";
  var WIDGET_ROLES = ["button", "link", "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "checkbox", "radio",
    "switch", "option", "treeitem", "combobox", "searchbox", "textbox", "slider", "spinbutton"];
  var SELECTOR = [
    "a[href]", "area[href]", "button", "input:not([type=hidden])", "select", "textarea", "summary",
    "audio[controls]", "video[controls]", "[onclick]", "[tabindex]:not([tabindex='-1'])",
    "[contenteditable='true']", "[contenteditable='']", "[contenteditable='plaintext-only']"
  ].concat(WIDGET_ROLES.map(function (r) { return "[role~=" + r + "]"; })).join(",");
  // roles whose name may come from their content (accname 1.2 "name from content")
  var NAME_FROM_CONTENT = ["button", "cell", "checkbox", "columnheader", "gridcell", "heading", "link", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "radio", "row", "rowheader", "switch", "tab", "tooltip", "treeitem",
    "summary"];
  var ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
  var TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
  var NUM = ONES.slice(0, 11);
  var STOP = {};
  ("a an the and or nor but of to for from by with at as into onto via per is are was were be been being it its " +
   "this that these those your you yours my me mine our ours we us i he she they them their his her here there " +
   "what which who whom whose than then so if also just very please s t").split(" ").forEach(function (w) { STOP[w] = 1; });
  // words that describe the widget rather than what it does; dropped when anything else is left
  var WEAK = {};
  "button btn link icon click tap here opens window external".split(" ").forEach(function (w) { WEAK[w] = 1; });
  var PARTICLE = { "in": 1, out: 1, up: 1, down: 1, on: 1, off: 1, over: 1 };
  // children laid out as boxes are separated by a space ("Issues" + "153", "Sign in" + <img>)
  var BLOCK = /^(block|flex|grid|list-item|table|table-row|table-cell|flow-root|inline-block|inline-flex|inline-grid)/;
  var REPLACED = { img: 1, svg: 1, input: 1, select: 1, textarea: 1, button: 1, video: 1, canvas: 1 };

  // ------------------------------------------------------------------ small DOM helpers (shadow-DOM aware)
  function parentOf(n) { return n.parentNode && n.parentNode.nodeType === 11 ? n.parentNode.host : (n.assignedSlot || n.parentElement); }
  function composedContains(a, b) { for (var n = b; n; n = parentOf(n)) if (n === a) return true; return false; }
  function composedClosest(el, sel) { for (var n = el; n; n = parentOf(n)) if (n.nodeType === 1 && n.matches(sel)) return n; return null; }
  function winOf(el) { return el.ownerDocument.defaultView; }
  function style(el) { return winOf(el).getComputedStyle(el); }
  function attr(el, a) { var v = el.getAttribute && el.getAttribute(a); return v == null ? null : v; }
  function nonEmpty(s) { return s != null && s.replace(/\s+/g, "") !== ""; }
  function byId(el, id) {
    var r = el.getRootNode ? el.getRootNode() : el.ownerDocument;
    return (r.getElementById ? r.getElementById(id) : null) || el.ownerDocument.getElementById(id);
  }
  function tag(el) { return el.tagName ? el.tagName.toLowerCase() : ""; }

  /** Explicit role if valid, else the implicit HTML-AAM role (subset that matters for naming and kinds). */
  function roleOf(el) {
    var r = (attr(el, "role") || "").trim().split(/\s+/)[0];
    if (r && r !== "none" && r !== "presentation") return r;
    var t = tag(el), type = (attr(el, "type") || "text").toLowerCase();
    if ((t === "a" || t === "area") && el.hasAttribute("href")) return "link";
    if (t === "button") return "button";
    if (t === "summary") return "summary";
    if (t === "select") return el.multiple || el.size > 1 ? "listbox" : "combobox";
    if (t === "textarea") return "textbox";
    if (t === "option") return "option";
    if (t === "img") return attr(el, "alt") === "" ? "presentation" : "img";
    if (/^h[1-6]$/.test(t)) return "heading";
    if (t === "td") return "cell";
    if (t === "th") return "columnheader";
    if (t === "input") {
      if (["button", "submit", "reset", "image"].indexOf(type) >= 0) return "button";
      if (type === "checkbox" || type === "radio") return type;
      if (type === "range") return "slider";
      if (type === "number") return "spinbutton";
      if (type === "search") return el.hasAttribute("list") ? "combobox" : "searchbox";
      return el.hasAttribute("list") ? "combobox" : "textbox";
    }
    if (el.isContentEditable && (!el.parentElement || !el.parentElement.isContentEditable)) return "textbox";
    return r || "";
  }

  function isHidden(el) {
    if (el.nodeType !== 1) return false;
    if (composedClosest(el, "[hidden]:not([hidden=until-found]), [aria-hidden='true'], [inert]")) return true;
    var s = style(el);
    return s.display === "none" || s.visibility === "hidden" || s.visibility === "collapse";
  }

  function cssContent(el, pseudo) {
    var ps = winOf(el).getComputedStyle(el, pseudo), c = ps.content;
    if (!c || c === "none" || c === "normal" || ps.display === "none" || ps.visibility === "hidden") return "";
    var m = c.match(/^"(.*)"$/) || c.match(/"(.*)"\s*\/\s*"(.*)"$/);
    if (!m) return "";
    var txt = m.length > 2 ? m[2] : m[1];
    return /^[\s -]*$/.test(txt) ? "" : txt;    // private-use icon-font glyphs say nothing
  }

  // ------------------------------------------------------------------ accessible name (W3C accname 1.2)
  function embeddedValue(el, role) {
    var t = tag(el);
    if (role === "textbox" || role === "searchbox") return t === "input" || t === "textarea" ? el.value : el.textContent;
    if (role === "combobox" || role === "listbox") {
      if (t === "select") return Array.prototype.filter.call(el.options, function (o) { return o.selected; }).map(function (o) { return o.textContent; }).join(" ");
      if (t === "input") return el.value;
      var sel = el.querySelector("[aria-selected='true']"); return sel ? sel.textContent : "";
    }
    if (role === "slider" || role === "spinbutton" || role === "progressbar" || role === "scrollbar")
      return attr(el, "aria-valuetext") || attr(el, "aria-valuenow") || (t === "input" ? el.value : "");
    return null;
  }

  function nameOf(node, ctx) {
    ctx = ctx || {};
    var visited = ctx.visited || (ctx.visited = new Set());
    if (node.nodeType === 3) return node.data;
    if (node.nodeType !== 1 || visited.has(node)) return "";
    visited.add(node);
    var el = node, t = tag(el), role = roleOf(el);
    if (t === "script" || t === "style" || t === "noscript" || t === "template") return "";
    // 2A: hidden, unless we got here through aria-labelledby / aria-describedby
    if (!ctx.referenced && isHidden(el)) return "";
    // 2B: aria-labelledby (not followed when already inside a labelledby traversal)
    var lb = attr(el, "aria-labelledby");
    if (lb && !ctx.inLabelledBy) {
      var parts = lb.trim().split(/\s+/).map(function (id) {
        var ref = byId(el, id);
        return ref ? nameOf(ref, { visited: visited, inLabelledBy: true, referenced: true, recursing: true }).trim() : "";
      }).filter(Boolean);
      if (parts.length) return parts.join(" ");
    }
    // 2C-2E: embedded control while computing another control's name from a label or content
    var embedded = ctx.recursing ? embeddedValue(el, role) : null;
    if (embedded != null && ctx.recursing && !ctx.isRoot) {
      var al0 = attr(el, "aria-label");
      if (nonEmpty(al0)) return al0.trim();
      return embedded;
    }
    // 2D: aria-label
    var al = attr(el, "aria-label");
    if (nonEmpty(al)) return al.trim();
    // 2E: native host-language label (HTML-AAM)
    if (role !== "presentation" && role !== "none") {
      var nat = nativeName(el, t, ctx);
      if (nonEmpty(nat)) return nat.trim();
    }
    // 2F: name from content
    if (NAME_FROM_CONTENT.indexOf(role) >= 0 || ctx.recursing || t === "label" || t === "legend" || t === "caption") {
      var txt = contentName(el, ctx);
      if (nonEmpty(txt)) return txt;
    }
    // 2I: tooltip, then placeholder (HTML-AAM order for inputs)
    var title = attr(el, "title");
    if (nonEmpty(title)) return title.trim();
    if (t === "input" || t === "textarea") {
      var ph = attr(el, "placeholder") || attr(el, "aria-placeholder");
      if (nonEmpty(ph)) return ph.trim();
    }
    var aph = attr(el, "aria-placeholder");
    if (nonEmpty(aph)) return aph.trim();
    return "";
  }

  function labelText(lbl, ctx) {
    return nameOf(lbl, { visited: ctx.visited, recursing: true, referenced: false, inLabelledBy: ctx.inLabelledBy });
  }

  function nativeName(el, t, ctx) {
    var type = (attr(el, "type") || "").toLowerCase();
    if (t === "input" || t === "textarea" || t === "select" || t === "meter" || t === "progress" || t === "output") {
      if (t === "input" && type === "image") return attr(el, "alt") || attr(el, "value") || attr(el, "title") || "Submit";
      if (el.labels && el.labels.length) {
        var s = Array.prototype.map.call(el.labels, function (l) { return labelText(l, ctx).trim(); }).filter(Boolean).join(" ");
        if (s) return s;
      }
      if (t === "input" && (type === "submit" || type === "reset" || type === "button")) {
        var v = attr(el, "value");
        if (nonEmpty(v)) return v;
        return type === "submit" ? "Submit" : type === "reset" ? "Reset" : "";
      }
      return "";
    }
    if (t === "img" || t === "area") return attr(el, "alt") || "";
    if (t === "svg") {
      var st = Array.prototype.filter.call(el.children, function (c) { return tag(c) === "title"; })[0];
      return st ? st.textContent : "";
    }
    if (t === "fieldset") { var lg = el.querySelector(":scope > legend"); return lg ? labelText(lg, ctx) : ""; }
    if (t === "figure") { var fc = el.querySelector(":scope > figcaption"); return fc ? labelText(fc, ctx) : ""; }
    if (t === "table") { var cp = el.querySelector(":scope > caption"); return cp ? labelText(cp, ctx) : ""; }
    return "";
  }

  function contentName(el, ctx) {
    var out = [cssContent(el, "::before")];
    var kids = shadowOf(el) ? shadowOf(el).childNodes : (tag(el) === "slot" ? (el.assignedNodes().length ? el.assignedNodes() : el.childNodes) : el.childNodes);
    var sub = { visited: ctx.visited, recursing: true, referenced: !!(ctx.referenced && isHidden(el)), inLabelledBy: ctx.inLabelledBy };
    Array.prototype.forEach.call(kids, function (c) {
      var s = nameOf(c, sub);
      if (c.nodeType === 1 && (REPLACED[tag(c)] || BLOCK.test(style(c).display))) s = " " + s + " ";
      if (c.nodeType === 1 && tag(c) === "br") s = " ";
      out.push(s);
    });
    out.push(cssContent(el, "::after"));
    return out.join("").replace(/[\s ]+/g, " ").trim();
  }

  /** Accessible name of an element, flattened to one line. With loose=true, an element whose role does not take
      its name from content (a clickable <div>, say) still falls back to its text, since we need something to say. */
  function accessibleName(el, loose) {
    var n = nameOf(el, { isRoot: true });
    if (!nonEmpty(n) && loose && ["textbox", "searchbox", "combobox", "listbox", "spinbutton", "slider"].indexOf(roleOf(el)) < 0)
      n = contentName(el, { visited: new Set([el]), recursing: true });
    return n.replace(/[\s ]+/g, " ").trim();
  }

  // ------------------------------------------------------------------ short spoken labels
  function spell(n) {
    if (n < 20) return ONES[n];
    if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? " " + ONES[n % 10] : "");
    if (n < 1000) return ONES[Math.floor(n / 100)] + " hundred" + (n % 100 ? " " + spell(n % 100) : "");
    return String(n).split("").map(function (d) { return ONES[+d]; }).join(" ");
  }

  /** Lowercase, ASCII-fold, drop punctuation and emoji; digits stay (shortLabel decides). Matches semg.text charset. */
  function norm(t) {
    return (t || "").normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[’‘]/g, "'").toLowerCase()
      .replace(/[^a-z0-9' ]+/g, " ").replace(/(^|\s)'+|'+(\s|$)/g, " ").replace(/\s+/g, " ").trim();
  }

  /** Turn an accessible name into a 1-2 word speakable label ("" when nothing speakable is left). */
  function shortLabel(name) {
    // counts like "12.3k", "1,204" are volatile and not worth saying
    var s = (name || "").replace(/\b\d+(?:[.,]\d+)*\s?[kmb]\b/gi, " ").replace(/\b\d{1,3}(?:,\d{3})+\b/g, " ")
      .replace(/\b\d+[.,]\d+\b/g, " ").replace(/([A-Za-z]{3,})(\d)/g, "$1 $2").replace(/(\d)([A-Za-z]{3,})/g, "$1 $2");   // "7Branches", not "611c616"
    var toks = norm(s).split(" ").filter(function (w, i, a) { return w && a.indexOf(w) === i; });   // "json - JSON ..."
    if (!toks.length) return "";
    var words = toks.filter(function (w) { return !/\d/.test(w); });
    var nums = toks.filter(function (w) { return /^\d+$/.test(w); });
    if (!words.length) return nums.length && +nums[0] < 1000 ? spell(+nums[0]) : "";
    var content = [], i;
    for (i = 0; i < toks.length && content.length < 2; i++) {
      var w = toks[i];
      if (/\d/.test(w)) continue;
      if (STOP[w] || PARTICLE[w] || NUMWORD[w] || w.length < 2) continue;
      content.push({ w: w, i: i });
    }
    var strong = content.filter(function (c) { return !WEAK[c.w]; });
    if (strong.length) {
      // re-collect up to two non-weak content words
      strong = [];
      for (i = 0; i < toks.length && strong.length < 2; i++)
        if (!/\d/.test(toks[i]) && !STOP[toks[i]] && !PARTICLE[toks[i]] && !NUMWORD[toks[i]] && !WEAK[toks[i]] && toks[i].length > 1) strong.push({ w: toks[i], i: i });
      content = strong;
    }
    if (!content.length) return words.filter(function (w) { return w.length > 1; }).slice(0, 2).join(" ");       // "About us" -> "about us"
    var out = [content[0].w];
    var next = toks[content[0].i + 1];
    var after = toks[content[0].i + 2];
    // phrasal verbs: "sign in", "log out of your account"; but "share on twitter" -> "share twitter"
    if (content[0].i === 0 && next && PARTICLE[next] && (!after || STOP[after])) out.push(next);
    else if (content.length > 1) out.push(content[1].w);
    else if (/^\s*[A-Za-z]+\s+\d{1,2}\s*$/.test(name) && toks[0] === content[0].w)
      out.push(spell(+toks[1]));                                  // "page 2" -> "page two" (not "6 comments")
    return out.join(" ").split(" ").slice(0, 3).join(" ");
  }

  // ------------------------------------------------------------------ candidates and visibility
  // open shadow roots; inside the extension chrome.dom also opens closed ones
  var CHROME_DOM = typeof chrome !== "undefined" && chrome.dom && chrome.dom.openOrClosedShadowRoot;
  function shadowOf(e) {
    if (e.shadowRoot) return e.shadowRoot;
    if (CHROME_DOM && e.tagName.indexOf("-") > 0) { try { return chrome.dom.openOrClosedShadowRoot(e); } catch (x) { return null; } }
    return null;
  }
  function collect(rootNode, sel, out) {
    Array.prototype.forEach.call(rootNode.querySelectorAll(sel), function (e) { out.push(e); });
    Array.prototype.forEach.call(rootNode.querySelectorAll("*"), function (e) { var s = shadowOf(e); if (s) collect(s, sel, out); });
    return out;
  }
  function allElements(rootNode, out, cap) {
    var list = rootNode.querySelectorAll("*");
    for (var i = 0; i < list.length && out.length < cap; i++) {
      out.push(list[i]);
      var s = shadowOf(list[i]);
      if (s) allElements(s, out, cap);
    }
    return out;
  }
  function deepHit(doc, x, y) {
    var hit = doc.elementFromPoint(x, y);
    while (hit && shadowOf(hit)) {
      var inner = shadowOf(hit).elementFromPoint(x, y);
      if (!inner || inner === hit) break;
      hit = inner;
    }
    return hit;
  }

  /** Returns the element's rect when a person can see it (and, in the viewport, reach it), else null. */
  /** Box of an element; a display:contents element has no box of its own, so use the union of its children's. */
  function boxOf(el, win) {
    if (win.getComputedStyle(el).display !== "contents") return { r: el.getBoundingClientRect(), probe: el };
    var L = Infinity, T = Infinity, R = -Infinity, B = -Infinity, probe = null;
    Array.prototype.forEach.call(el.children, function (c) {
      var b = boxOf(c, win).r;
      if (b.width < 1 || b.height < 1) return;
      probe = probe || c;
      L = Math.min(L, b.left); T = Math.min(T, b.top); R = Math.max(R, b.right); B = Math.max(B, b.bottom);
    });
    if (!probe) return { r: { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }, probe: el };
    return { r: { left: L, top: T, right: R, bottom: B, width: R - L, height: B - T, x: L, y: T }, probe: probe };
  }

  function rendered(el, win, scope) {
    var box = boxOf(el, win), r = box.r, vis = box.probe;
    if (r.width < 4 || r.height < 4) return null;
    var inView = !(r.bottom <= 0 || r.right <= 0 || r.top >= win.innerHeight || r.left >= win.innerWidth);
    if (scope !== "page" && !inView) return null;
    if (r.right <= 0 || r.left >= win.document.documentElement.scrollWidth) return null;
    if (vis.checkVisibility) { if (!vis.checkVisibility({ opacityProperty: true, visibilityProperty: true })) return null; }
    else {
      var s = win.getComputedStyle(vis);
      if (s.visibility !== "visible" || s.display === "none" || parseFloat(s.opacity) < 0.05) return null;
    }
    var cs = win.getComputedStyle(el);
    if (cs.clipPath === "inset(50%)" || /rect\(0(px)?,? 0(px)?,? 0(px)?,? 0(px)?\)/.test(cs.clip)) return null;
    if (composedClosest(el, "[aria-hidden='true'], [inert]")) return null;
    if ((el.matches && el.matches(":disabled")) || attr(el, "aria-disabled") === "true") return null;
    if (!inView) return r;                                        // below the fold: cannot hit-test, trust the styles
    var doc = el.ownerDocument, W = win.innerWidth - 1, H = win.innerHeight - 1;
    var pts = [[r.left + r.width / 2, r.top + r.height / 2], [r.left + Math.min(r.width / 2, 12), r.top + r.height / 2],
               [r.left + r.width / 2, r.top + Math.min(r.height / 2, 6)]];
    for (var i = 0; i < pts.length; i++) {
      var x = Math.min(W, Math.max(0, pts[i][0])), y = Math.min(H, Math.max(0, pts[i][1]));
      var hit = deepHit(doc, x, y);
      if (!hit || composedContains(el, hit) || composedContains(hit, el)) return r;
      var lbl = composedClosest(hit, "label");
      if (lbl && lbl.control === el) return r;
    }
    return null;                                                  // covered by something else
  }

  /** Kind used for badges and grammar: link, button, field, checkbox, radio, menu, tab, ... */
  function kind(el) {
    var role = roleOf(el);
    if (role === "textbox" || role === "searchbox" || role === "spinbutton") return role === "searchbox" ? "searchbox" : "field";
    if (role === "combobox") return tag(el) === "select" ? "menu" : "combobox";
    if (role === "listbox") return "menu";
    if (role === "summary") return "button";
    if (["link", "button", "checkbox", "radio", "tab", "menuitem", "switch", "option", "treeitem", "slider"].indexOf(role) >= 0) return role;
    if (role === "menuitemcheckbox" || role === "menuitemradio") return "menuitem";
    if (tag(el) === "input") return "field";
    return "button";
  }

  function isSemantic(el) {
    var t = tag(el);
    if (["a", "area", "button", "input", "select", "textarea", "summary", "audio", "video"].indexOf(t) >= 0) return true;
    if (el.isContentEditable) return true;
    return WIDGET_ROLES.indexOf(roleOf(el)) >= 0 && el.hasAttribute("role");
  }

  /**
   * Returns targets in reading order:
   *   [{n, el, name, label, display, kind, rect, source}]
   * n: badge number. name: full accessible name. label: spoken words, unique on the page ("reply two").
   * display: the same with digits ("reply 2"). source: "semantic" | "handler" | "cursor".
   * opts.scope: "viewport" (default; what is on screen now) or "page" (whole document, for evaluation).
   */
  function detect(doc, opts) {
    doc = doc || document; opts = opts || {};
    var win = doc.defaultView, scope = opts.scope || "viewport", seen = new Set(), found = [];
    function consider(el, source) {
      if (seen.has(el)) return; seen.add(el);
      var t = tag(el);
      if (t === "a" && !el.hasAttribute("href") && source === "semantic" && !el.hasAttribute("role")) return;
      var r = rendered(el, win, scope), target = el;
      if (!r && t === "input" && el.labels && el.labels.length) {
        // custom checkboxes/radios: the real input is transparent or tiny and its <label> is what people see
        for (var i = 0; i < el.labels.length && !r; i++) { r = rendered(el.labels[i], win, scope); if (r) target = el.labels[i]; }
        if (r) seen.add(target);
      }
      if (!r) return;
      found.push({ el: el, click: target, rect: r, source: source });
    }
    var cands = collect(doc, SELECTOR, []);
    cands.forEach(function (el) {
      var t = tag(el), semantic = isSemantic(el);
      if (!semantic && el.hasAttribute("tabindex") && !el.hasAttribute("onclick")) {
        // tabindex alone marks many scroll regions and dialogs; require a pointer cursor too
        if (style(el).cursor !== "pointer") return;
      }
      consider(el, semantic ? "semantic" : "handler");
    });
    // elements that only reveal themselves through a pointer cursor (click handlers added in script)
    var all = doc.body ? allElements(doc.body, [], 6000) : [];
    for (var i = 0; i < all.length; i++) {
      var el = all[i], t = el.tagName;
      if (seen.has(el) || !(t === "DIV" || t === "SPAN" || t === "LI" || t === "IMG" || t === "svg" || t === "SVG" || t === "I" || t === "TD")) continue;
      if (style(el).cursor !== "pointer") continue;
      var p = parentOf(el);
      if (p && p.nodeType === 1 && style(p).cursor === "pointer") continue;   // keep the outermost
      if (composedClosest(el, "label")) continue;                              // the label's control is the target
      consider(el, "cursor");
    }
    // Nesting: a native control inside a clickable card is its own target; a cursor/handler element inside
    // another target is part of that target.
    var set = new Set(found.map(function (f) { return f.el; }));
    found = found.filter(function (f) {
      if (f.source === "semantic") {
        for (var a = parentOf(f.el); a; a = parentOf(a)) {
          if (set.has(a) && isSemantic(a)) {
            var ft = tag(f.el), at = tag(a);
            // <a><button>x</button></a> or <button><span role=button>: one control, keep the outer
            if ((at === "a" || at === "button") && (ft === "button" || ft === "a" || !isSemanticTag(ft))) return false;
          }
        }
        return true;
      }
      for (var b = parentOf(f.el); b; b = parentOf(b)) if (set.has(b)) return false;
      // a pointer-cursor card that wraps real links/buttons: the controls inside are the targets
      if (f.source === "cursor" && found.some(function (g) { return g !== f && g.source === "semantic" && composedContains(f.el, g.el); })) return false;
      return true;
    });
    found.sort(function (a, b) { return (Math.round(a.rect.top / 8) - Math.round(b.rect.top / 8)) || (a.rect.left - b.rect.left); });
    var targets = found.map(function (f, i) {
      var name = accessibleName(f.el, true), k = kind(f.el);
      return { n: i + 1, el: f.el, click: f.click, name: name, base: spokenBase(f.el, name, k), kind: k, rect: f.rect, source: f.source };
    });
    numberDuplicates(targets);
    return targets;
  }
  /** Short label a person would say. Voice-control users say what they see (WCAG 2.5.3 "label in name"), so when an
      aria-label/labelledby replaces visible text, the visible text wins ("Star 54.4k" over "You must be signed in to
      star a repository"); icon-only controls keep their accessible name. Bare numbers ("[1]", "2") would clash with
      badge numbers, so they become "<kind> <number>". */
  function spokenBase(el, name, k) {
    var base = shortLabel(name);
    if ((el.hasAttribute("aria-label") || el.hasAttribute("aria-labelledby")) && /^(link|button|tab|menuitem|option|treeitem|switch)$/.test(k)) {
      var vis = shortLabel(el.innerText || "");
      if (vis) base = vis;
    }
    if (base && base.split(" ").every(function (w) { return NUMWORD[w]; })) base = k + " " + base;
    return base || k;
  }
  var NUMWORD = {};
  ONES.concat(TENS, ["hundred"]).forEach(function (w) { if (w) NUMWORD[w] = 1; });
  function isSemanticTag(t) { return ["a", "area", "button", "input", "select", "textarea", "summary"].indexOf(t) >= 0; }

  /** Adds label/display: duplicates of a base label are numbered in reading order ("reply one", "reply two"). */
  function numberDuplicates(targets) {
    var count = {}, seenN = {};
    targets.forEach(function (t) { count[t.base] = (count[t.base] || 0) + 1; });
    var taken = {};
    targets.forEach(function (t) { if (count[t.base] === 1) taken[t.base] = 1; });
    targets.forEach(function (t) {
      if (count[t.base] === 1) { t.label = t.base; t.display = t.base; return; }
      var k = seenN[t.base] = (seenN[t.base] || 0) + 1;
      var lab = t.base + " " + spell(k);
      while (taken[lab]) { k = seenN[t.base] = seenN[t.base] + 1; lab = t.base + " " + spell(k); }
      taken[lab] = 1;
      t.label = lab; t.display = t.base + " " + k;
    });
    return targets;
  }

  // ------------------------------------------------------------------ page lexicon
  var COMMANDS = {
    always: ["afono", "sleep", "show", "hide", "click", "open", "close", "back", "next", "stop", "zoom", "in", "out"],
    scroll: ["scroll", "up", "down", "page", "top", "bottom"],
    field: ["enter", "tab", "undo", "copy", "paste", "select", "delete", "send"],
    search: ["search"],
    dialog: ["yes", "no", "cancel", "done"]
  };
  function numberWords(maxN) {
    var w = new Set();
    for (var i = 1; i <= maxN; i++) spell(i).split(" ").forEach(function (x) { w.add(x); });
    return Array.from(w);
  }

  /**
   * The words the sEMG decoder needs for this page.
   * Returns {labels, commands, numbers, all}; labels = distinct words of the target labels ("words on screen"),
   * commands = the Tier 1 words that apply here, numbers = words needed to say every badge number.
   * opts.doc lets the caller say whether the page scrolls / has a dialog; opts.maxBadge caps badge numbers.
   */
  function lexicon(targets, opts) {
    opts = opts || {};
    var doc = opts.doc, labels = new Set(), cmds = new Set(COMMANDS.always);
    targets.forEach(function (t) { t.label.split(" ").forEach(function (w) { if (w) labels.add(w); }); });
    var scrolls = !doc || doc.documentElement.scrollHeight > doc.defaultView.innerHeight + 4;
    if (scrolls) COMMANDS.scroll.forEach(function (w) { cmds.add(w); });
    if (targets.some(function (t) { return t.kind === "field" || t.kind === "searchbox" || t.kind === "combobox"; }))
      COMMANDS.field.forEach(function (w) { cmds.add(w); });
    if (targets.some(function (t) { return t.kind === "searchbox" || /search/.test(t.label); })) cmds.add("search");
    if (doc && doc.querySelector("dialog[open], [role=dialog]:not([aria-hidden=true]), [role=alertdialog]"))
      COMMANDS.dialog.forEach(function (w) { cmds.add(w); });
    var nums = numberWords(Math.min(targets.length, opts.maxBadge || 150));
    var all = new Set();
    [labels, cmds, nums].forEach(function (s) { s.forEach(function (w) { all.add(w); }); });
    return { labels: Array.from(labels), commands: Array.from(cmds), numbers: nums, all: Array.from(all) };
  }

  /** Backward-compatible flat word list (what Nullpoint's "page" vocabulary adds to its command words). */
  function vocabulary(targets, opts) { return lexicon(targets, opts).all; }

  var api = { detect: detect, vocabulary: vocabulary, lexicon: lexicon, accessibleName: accessibleName,
    shortLabel: shortLabel, numberDuplicates: numberDuplicates, spell: spell, norm: norm, roleOf: roleOf, NUM: NUM,
    SELECTOR: SELECTOR, version: "dom-v1.1" };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.AfonoDOM = api;
})(typeof window !== "undefined" ? window : this);
