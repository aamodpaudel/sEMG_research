/* Afono model glyphs: the φ ring and bar morph into an eye, lips or a browser window.
   All shapes share 8 cubic segments, so SVG <animate> can tween them directly. */
(function () {
  "use strict";
  var SHAPES = {"ring": "M48.00 26.00 C53.83 26.00 59.43 28.32 63.56 32.44 C67.68 36.57 70.00 42.17 70.00 48.00 C70.00 53.83 67.68 59.43 63.56 63.56 C59.43 67.68 53.83 70.00 48.00 70.00 C42.17 70.00 36.57 67.68 32.44 63.56 C28.32 59.43 26.00 53.83 26.00 48.00 C26.00 42.17 28.32 36.57 32.44 32.44 C36.57 28.32 42.17 26.00 48.00 26.00 Z", "eye": "M48.00 34.36 C58.21 35.26 68.00 38.15 75.22 41.40 C82.44 44.17 86.50 46.37 86.50 48.00 C86.50 49.63 82.44 51.83 75.22 54.60 C68.00 57.85 58.21 60.74 48.00 61.64 C37.79 60.74 28.00 57.85 20.78 54.60 C13.56 51.83 9.50 49.63 9.50 48.00 C9.50 46.37 13.56 44.17 20.78 41.40 C28.00 38.15 37.79 35.26 48.00 34.36 Z", "lips": "M48.00 41.40 C58.79 38.46 69.15 38.20 76.78 40.94 C84.41 43.47 88.70 45.91 88.70 48.00 C88.70 50.35 84.41 53.10 76.78 55.96 C69.15 59.05 58.79 61.30 48.00 61.64 C37.21 61.30 26.85 59.05 19.22 55.96 C11.59 53.10 7.30 50.35 7.30 48.00 C7.30 45.91 11.59 43.47 19.22 40.94 C26.85 38.20 37.21 38.46 48.00 41.40 Z", "window": "M48.00 24.00 C68.75 24.00 77.86 23.14 81.36 26.64 C84.86 30.14 84.00 34.65 84.00 48.00 C84.00 61.35 84.86 65.86 81.36 69.36 C77.86 72.86 68.75 72.00 48.00 72.00 C27.25 72.00 18.14 72.86 14.64 69.36 C11.14 65.86 12.00 61.35 12.00 48.00 C12.00 34.65 11.14 30.14 14.64 26.64 C18.14 23.14 27.25 24.00 48.00 24.00 Z"};
  var BAR = { ring: [48, 8, 48, 88, 9], eye: [48, 48, 48, 48.01, 17], lips: [20, 49.5, 76, 49.5, 3.5], window: [26, 58, 54, 58, 9] };
  var RING_W = { ring: 9, eye: 5.5, lips: 5.5, window: 5.5 };
  var NS = "http://www.w3.org/2000/svg";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function anim(el, attr, from, to, dur) {
    var a = document.createElementNS(NS, "animate");
    a.setAttribute("attributeName", attr); a.setAttribute("values", from + ";" + to);
    a.setAttribute("dur", dur + "s"); a.setAttribute("fill", "freeze"); a.setAttribute("begin", "indefinite");
    a.setAttribute("calcMode", "spline"); a.setAttribute("keyTimes", "0;1"); a.setAttribute("keySplines", "0.7 0 0.2 1");
    el.appendChild(a); return a;
  }
  function setup(svg) {
    var shape = svg.dataset.shape, ring = svg.querySelector(".m-ring"), bar = svg.querySelector(".m-bar");
    var extra = null;
    if (shape === "window") {
      extra = document.createElementNS(NS, "line");
      [["x1", 20], ["y1", 35], ["x2", 76], ["y2", 35], ["stroke", "var(--teal)"], ["stroke-width", 3], ["stroke-linecap", "round"], ["opacity", 0]]
        .forEach(function (kv) { extra.setAttribute(kv[0], kv[1]); });
      svg.insertBefore(extra, bar);
    }
    function set(name) {
      ring.setAttribute("d", SHAPES[name]); ring.setAttribute("stroke-width", RING_W[name]);
      var b = BAR[name]; ["x1", "y1", "x2", "y2", "stroke-width"].forEach(function (k, i) { bar.setAttribute(k, b[i]); });
      if (extra) extra.setAttribute("opacity", name === "window" ? 0.55 : 0);
    }
    function build(a, b, dur) {
      var list = [anim(ring, "d", SHAPES[a], SHAPES[b], dur), anim(ring, "stroke-width", RING_W[a], RING_W[b], dur)];
      ["x1", "y1", "x2", "y2", "stroke-width"].forEach(function (k, i) { list.push(anim(bar, k, BAR[a][i], BAR[b][i], dur)); });
      if (extra) list.push(anim(extra, "opacity", a === "window" ? 0.55 : 0, b === "window" ? 0.55 : 0, dur));
      return list;
    }
    set("ring");
    if (reduce || !svg.beginElement && !ring.animate) { set(shape); return; }
    var fwd = build("ring", shape, 0.95), back = build(shape, "ring", 0.75), state = "ring", timer = null, visible = false;
    function go(list, next) { list.forEach(function (a) { try { a.beginElement(); } catch (e) {} }); state = next; }
    function cycle() {
      clearTimeout(timer);
      if (!visible) return;
      if (state === "ring") { go(fwd, shape); timer = setTimeout(cycle, 3400); }
      else { go(back, "ring"); timer = setTimeout(cycle, 1500); }
    }
    new IntersectionObserver(function (es) {
      var v = es[0].isIntersecting;
      if (v && !visible) { visible = true; timer = setTimeout(cycle, state === "ring" ? 350 : 2600); }
      if (!v) { visible = false; clearTimeout(timer); }
    }, { threshold: 0.4 }).observe(svg);
    svg.closest(".model-card").addEventListener("mouseenter", function () {
      if (state === "ring") { clearTimeout(timer); go(fwd, shape); timer = setTimeout(cycle, 3400); }
    });
  }
  document.querySelectorAll("svg.morph").forEach(setup);
})();
