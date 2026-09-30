/* Afono model glyphs: the phi ring and bar morph into an eye, lips or a browser window.
   All shapes share 8 cubic segments, so SVG <animate> can tween them directly. */
(function () {
  "use strict";
  var SHAPES = {"ring": "M48.00 26.00 C53.83 26.00 59.43 28.32 63.56 32.44 C67.68 36.57 70.00 42.17 70.00 48.00 C70.00 53.83 67.68 59.43 63.56 63.56 C59.43 67.68 53.83 70.00 48.00 70.00 C42.17 70.00 36.57 67.68 32.44 63.56 C28.32 59.43 26.00 53.83 26.00 48.00 C26.00 42.17 28.32 36.57 32.44 32.44 C36.57 28.32 42.17 26.00 48.00 26.00 Z", "eye": "M48.00 30.84 C57.92 31.82 67.43 35.58 74.45 39.90 C81.46 43.61 85.40 46.27 85.40 48.00 C85.40 49.73 81.46 52.39 74.45 56.10 C67.43 60.42 57.92 64.18 48.00 65.16 C38.08 64.18 28.57 60.42 21.55 56.10 C14.54 52.39 10.60 49.73 10.60 48.00 C10.60 46.27 14.54 43.61 21.55 39.90 C28.57 35.58 38.08 31.82 48.00 30.84 Z", "lips": "M48.00 41.86 C58.21 36.65 68.00 37.28 75.22 40.53 C82.44 43.47 86.50 46.01 86.50 48.00 C86.50 50.57 82.44 53.92 75.22 57.83 C68.00 62.07 58.21 65.17 48.00 65.60 C37.79 65.17 28.00 62.07 20.78 57.83 C13.56 53.92 9.50 50.57 9.50 48.00 C9.50 46.01 13.56 43.47 20.78 40.53 C28.00 37.28 37.79 36.65 48.00 41.86 Z", "window": "M48.00 24.00 C68.75 24.00 77.86 23.14 81.36 26.64 C84.86 30.14 84.00 34.65 84.00 48.00 C84.00 61.35 84.86 65.86 81.36 69.36 C77.86 72.86 68.75 72.00 48.00 72.00 C27.25 72.00 18.14 72.86 14.64 69.36 C11.14 65.86 12.00 61.35 12.00 48.00 C12.00 34.65 11.14 30.14 14.64 26.64 C18.14 23.14 27.25 24.00 48.00 24.00 Z"};
  // bar: x1, y1, x2, y2, stroke width, opacity
  var BAR = { ring: [48, 10, 48, 86, 6.5, 1], eye: [48, 45, 48, 51, 6.5, 0], lips: [21, 50.5, 75, 50.5, 3, 1], window: [26, 58, 54, 58, 6.5, 1] };
  var RING_W = { ring: 6.5, eye: 4, lips: 4, window: 4 };
  var PUPIL = { ring: 0, eye: 7, lips: 0, window: 0 };
  var TITLE = { ring: 0, eye: 0, lips: 0, window: 0.55 };
  var ATTRS = ["x1", "y1", "x2", "y2", "stroke-width", "opacity"];
  var NS = "http://www.w3.org/2000/svg";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function anim(el, attr, from, to, dur) {
    var a = document.createElementNS(NS, "animate");
    a.setAttribute("attributeName", attr); a.setAttribute("values", from + ";" + to);
    a.setAttribute("dur", dur + "s"); a.setAttribute("fill", "freeze"); a.setAttribute("begin", "indefinite");
    a.setAttribute("calcMode", "spline"); a.setAttribute("keyTimes", "0;1"); a.setAttribute("keySplines", "0.65 0 0.25 1");
    el.appendChild(a); return a;
  }
  function el(tag, attrs, before, parent) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    parent.insertBefore(e, before); return e;
  }
  function setup(svg) {
    var shape = svg.dataset.shape, ring = svg.querySelector(".m-ring"), bar = svg.querySelector(".m-bar");
    var pupil = el("circle", { cx: 48, cy: 48, r: 0, fill: "var(--signal)" }, null, svg);
    var title = el("line", { x1: 20, y1: 35, x2: 76, y2: 35, stroke: "var(--teal)", "stroke-width": 2.5, "stroke-linecap": "round", opacity: 0 }, bar, svg);
    function set(name) {
      ring.setAttribute("d", SHAPES[name]); ring.setAttribute("stroke-width", RING_W[name]);
      ATTRS.forEach(function (k, i) { bar.setAttribute(k, BAR[name][i]); });
      pupil.setAttribute("r", PUPIL[name]); title.setAttribute("opacity", TITLE[name]);
    }
    function build(a, b, dur) {
      var list = [anim(ring, "d", SHAPES[a], SHAPES[b], dur), anim(ring, "stroke-width", RING_W[a], RING_W[b], dur),
                  anim(pupil, "r", PUPIL[a], PUPIL[b], dur), anim(title, "opacity", TITLE[a], TITLE[b], dur)];
      ATTRS.forEach(function (k, i) { list.push(anim(bar, k, BAR[a][i], BAR[b][i], dur)); });
      return list;
    }
    set("ring");
    if (reduce || !ring.animate) { set(shape); return; }
    var fwd = build("ring", shape, 1.0), back = build(shape, "ring", 0.8), state = "ring", timer = null, visible = false;
    function go(list, next) { list.forEach(function (a) { try { a.beginElement(); } catch (e) {} }); state = next; }
    function cycle() {
      clearTimeout(timer);
      if (!visible) return;
      if (state === "ring") { go(fwd, shape); timer = setTimeout(cycle, 3600); }
      else { go(back, "ring"); timer = setTimeout(cycle, 1600); }
    }
    new IntersectionObserver(function (es) {
      var v = es[0].isIntersecting;
      if (v && !visible) { visible = true; timer = setTimeout(cycle, state === "ring" ? 350 : 2600); }
      if (!v) { visible = false; clearTimeout(timer); }
    }, { threshold: 0.4 }).observe(svg);
    (svg.closest(".tile, .model-card") || svg).addEventListener("mouseenter", function () {
      if (state === "ring") { clearTimeout(timer); go(fwd, shape); timer = setTimeout(cycle, 3600); }
    });
  }
  document.querySelectorAll("svg.morph").forEach(setup);
})();
