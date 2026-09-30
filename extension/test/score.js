/* Afono DOM Model evaluation helpers, run inside the page under test (headless Chrome via semg/dom_eval.py,
   or test/run.html in a normal browser). Needs dom-model.js and commands.js loaded first. */
(function (root) {
  "use strict";
  var D = root.AfonoDOM, C = root.AfonoCommands;

  function up(n) { return n.parentNode && n.parentNode.nodeType === 11 ? n.parentNode.host : n.parentElement; }
  function contains(a, b) { for (var n = b; n; n = up(n)) if (n === a) return true; return false; }
  function closest(el, sel) { for (var n = el; n; n = up(n)) if (n.nodeType === 1 && n.matches(sel)) return n; return null; }

  /** A CSS path that finds this element again in the same (static) document. */
  function cssPath(el) {
    if (el.getRootNode() !== el.ownerDocument) return null;              // inside a shadow root
    var parts = [];
    for (var n = el; n && n.nodeType === 1 && n !== el.ownerDocument.documentElement; n = n.parentElement) {
      if (n.id && /^[A-Za-z][\w-]*$/.test(n.id) && el.ownerDocument.querySelectorAll("#" + n.id).length === 1) {
        parts.unshift("#" + n.id); break;
      }
      var i = 1, s = n;
      while ((s = s.previousElementSibling)) if (s.tagName === n.tagName) i++;
      parts.unshift(n.tagName.toLowerCase() + ":nth-of-type(" + i + ")");
    }
    return parts.join(" > ");
  }

  function summary(targets, doc) {
    var lex = D.lexicon(targets, { doc: doc });
    var labels = targets.map(function (t) { return t.label; });
    var uniq = new Set(labels).size;
    // analysis: how big would the lexicon be with one-word labels (first word, then numbered)?
    var one = D.numberDuplicates(targets.map(function (t) { return { base: t.base.split(" ")[0], kind: t.kind }; }));
    var oneWords = new Set();
    one.forEach(function (t) { t.label.split(" ").forEach(function (w) { oneWords.add(w); }); });
    return {
      targets: targets.length,
      named: targets.filter(function (t) { return !!D.shortLabel(t.name); }).length,
      unique_labels: uniq,
      vocab_labels: lex.labels.length, vocab_commands: lex.commands.length, vocab_numbers: lex.numbers.length,
      vocab_all: lex.all.length, vocab_labels_one_word: oneWords.size,
      by_source: targets.reduce(function (m, t) { m[t.source] = (m[t.source] || 0) + 1; return m; }, {})
    };
  }

  function rows(targets) {
    return targets.map(function (t) {
      return { n: t.n, label: t.label, name: (t.name || "").slice(0, 100), kind: t.kind, source: t.source,
        path: cssPath(t.el), rect: [Math.round(t.rect.left), Math.round(t.rect.top), Math.round(t.rect.width), Math.round(t.rect.height)] };
    });
  }

  /** Match detections to ground-truth elements (one-to-one; nesting either way counts as the same control). */
  function match(targets, gtEls) {
    var used = new Set(), tp = [], fp = [];
    targets.forEach(function (t) {
      var hit = -1;
      for (var pass = 0; pass < 2 && hit < 0; pass++) {
        for (var i = 0; i < gtEls.length; i++) {
          var g = gtEls[i];
          if (!g || used.has(i)) continue;
          var same = g === t.el || g === t.click;
          var nested = contains(g, t.el) || contains(t.el, g);
          if (pass === 0 ? same : nested) { hit = i; break; }
        }
      }
      if (hit >= 0) { used.add(hit); tp.push([t, hit]); } else fp.push(t);
    });
    var fn = [];
    gtEls.forEach(function (g, i) { if (!used.has(i)) fn.push(i); });
    return { tp: tp, fp: fp, fn: fn };
  }

  /** DOM-0/DOM-0b fixture: data-gt = expected spoken label, data-name = expected accessible name. */
  function fixture(doc) {
    var found = D.detect(doc);
    var gt = [];
    (function walk(r) {
      r.querySelectorAll("[data-gt]").forEach(function (e) { gt.push(e); });
      r.querySelectorAll("*").forEach(function (e) { if (e.shadowRoot) walk(e.shadowRoot); });
    })(doc);
    var m = match(found, gt), labelOk = 0, misses = [];
    m.tp.forEach(function (p) {
      var want = gt[p[1]].getAttribute("data-gt");
      if (p[0].label === want) labelOk++; else misses.push("label: got '" + p[0].label + "' want '" + want + "'");
    });
    m.fp.forEach(function (t) { misses.push("false positive: " + t.label); });
    m.fn.forEach(function (i) { misses.push("missed: " + gt[i].getAttribute("data-gt")); });
    var named = [];
    (function walk(r) {
      r.querySelectorAll("[data-name]").forEach(function (e) { named.push(e); });
      r.querySelectorAll("*").forEach(function (e) { if (e.shadowRoot) walk(e.shadowRoot); });
    })(doc);
    var nameOk = 0;
    named.forEach(function (e) {
      var got = D.accessibleName(e), want = e.getAttribute("data-name");
      if (got === want) nameOk++; else misses.push("name: got '" + got + "' want '" + want + "'");
    });
    var cmd = [["click two", "news"], ["click sign in", "sign in"], ["open project", "open project"],
               ["click reply two", "reply two"], ["share three", "share three"], ["click download report", "download report"],
               ["scroll down", null], ["zoom in", null]];
    var cmdOk = cmd.filter(function (c) {
      var a = C.parse(c[0], found);
      var ok = c[1] ? (a.type === "click" && a.target.label === c[1]) : a.type !== "unknown";
      if (!ok) misses.push("command: " + c[0]);
      return ok;
    }).length;
    var s = summary(found, doc);
    return { targets_found: found.length, ground_truth: gt.length, true_positives: m.tp.length,
      precision: m.tp.length / Math.max(1, found.length), recall: m.tp.length / Math.max(1, gt.length),
      label_accuracy: labelOk / Math.max(1, m.tp.length), name_checks: named.length, name_accuracy: nameOk / Math.max(1, named.length),
      labels_unique: s.unique_labels === found.length, commands_ok: cmdOk, commands_total: cmd.length,
      vocabulary: D.lexicon(found, { doc: doc }), vocabulary_size: s.vocab_all, misses: misses, targets: rows(found) };
  }

  /** DOM-1 real page: stats for the viewport and the whole page, plus detections for inspection. */
  function page(doc, gtPaths) {
    var t0 = performance.now();
    var vp = D.detect(doc, { scope: "viewport" });
    var ms = performance.now() - t0;
    var full = D.detect(doc, { scope: "page" });
    var out = { viewport: summary(vp, doc), page: summary(full, doc), detect_ms: Math.round(ms),
      scroll_height: doc.documentElement.scrollHeight, targets: rows(vp),
      viewport_lexicon: D.lexicon(vp, { doc: doc }).labels };
    if (gtPaths) {
      var gtEls = gtPaths.map(function (p) { try { return doc.querySelector(p); } catch (e) { return null; } });
      var m = match(vp, gtEls);
      out.gt = { n_gt: gtPaths.length, missing_gt_selectors: gtEls.filter(function (g) { return !g; }).length,
        tp: m.tp.length, fp: m.fp.length, fn: m.fn.length,
        fp_items: m.fp.map(function (t) { return t.n + " " + t.label + " | " + (t.name || "").slice(0, 60); }),
        fn_items: m.fn.map(function (i) { return gtPaths[i]; }),
        tp_labels: m.tp.map(function (p) { return p[0].label; }) };
    }
    return out;
  }

  /** Broad candidate list for hand annotation: anything in the viewport that might be clickable, detected or not. */
  function candidates(doc) {
    var win = doc.defaultView, det = D.detect(doc), detSet = new Set(det.map(function (t) { return t.el; })), out = [];
    var sel = "a,button,input,select,textarea,summary,label,[role],[onclick],[tabindex],[contenteditable]";
    Array.prototype.forEach.call(doc.body.querySelectorAll("*"), function (el) {
      var r = el.getBoundingClientRect();
      if (r.width < 4 || r.height < 4 || r.bottom <= 0 || r.top >= win.innerHeight || r.right <= 0 || r.left >= win.innerWidth) return;
      var pointer = win.getComputedStyle(el).cursor === "pointer";
      var parentPointer = el.parentElement && win.getComputedStyle(el.parentElement).cursor === "pointer";
      if (!(el.matches(sel) || (pointer && !parentPointer))) return;
      if (el.checkVisibility && !el.checkVisibility({ opacityProperty: true, visibilityProperty: true })) return;
      out.push({ path: cssPath(el), tag: el.tagName.toLowerCase(), role: el.getAttribute("role") || "", detected: detSet.has(el),
        name: D.accessibleName(el, true).slice(0, 80), rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)] });
    });
    return out;
  }

  /** Draw numbered boxes over detections (or given rects) for screenshots. */
  function overlay(doc, items) {
    var box = doc.createElement("div");
    box.id = "__afono_overlay";
    box.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483647";
    items.forEach(function (it, i) {
      var d = doc.createElement("div"), r = it.rect;
      d.style.cssText = "position:fixed;border:2px solid " + (it.color || "#F0794A") + ";left:" + r[0] + "px;top:" + r[1] + "px;width:" + r[2] + "px;height:" + r[3] + "px";
      var b = doc.createElement("span");
      b.textContent = it.tag != null ? it.tag : String(i + 1);
      b.style.cssText = "position:absolute;left:-2px;top:-14px;font:bold 11px/13px sans-serif;background:" + (it.color || "#F0794A") + ";color:#fff;padding:0 2px";
      d.appendChild(b); box.appendChild(d);
    });
    doc.documentElement.appendChild(box);
  }

  root.AfonoEval = { fixture: fixture, page: page, candidates: candidates, overlay: overlay, cssPath: cssPath };
})(typeof window !== "undefined" ? window : this);
