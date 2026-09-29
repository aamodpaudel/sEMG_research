/* Afono CTC decoders (JS port of semg/decode.py). Works in the browser (window.AfonoDecode) and Node. */
(function (root) {
  "use strict";
  var NEG = -Infinity;

  function lse(a, b) {
    if (a === NEG) return b;
    if (b === NEG) return a;
    var m = a > b ? a : b;
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
  }

  function greedy(lp, T, V, chars) {
    var out = "", prev = -1;
    for (var t = 0; t < T; t++) {
      var o = t * V, best = 0, bv = lp[o];
      for (var v = 1; v < V; v++) if (lp[o + v] > bv) { bv = lp[o + v]; best = v; }
      if (best !== prev && best !== 0) out += chars[best];
      prev = best;
    }
    return out.split(" ").filter(Boolean).join(" ");
  }

  function buildLexicon(counts, chars) {
    var id = {};
    chars.forEach(function (c, i) { id[c] = i; });
    var words = Object.keys(counts).filter(function (w) {
      if (!w) return false;
      for (var i = 0; i < w.length; i++) if (w[i] === " " || !(w[i] in id)) return false;
      return true;
    }).sort();
    var total = 0;
    words.forEach(function (w) { total += (counts[w] || 0) + 1; });
    var rootNode = { c: new Map(), w: null }, logp = {};
    words.forEach(function (w) {
      logp[w] = Math.log(((counts[w] || 0) + 1) / total);
      var n = rootNode;
      for (var i = 0; i < w.length; i++) {
        var k = id[w[i]];
        if (!n.c.has(k)) n.c.set(k, { c: new Map(), w: null });
        n = n.c.get(k);
      }
      n.w = w;
    });
    return { root: rootNode, logp: logp, size: words.length };
  }

  /** Lexicon-constrained CTC prefix beam search. lp: Float32Array (T*V) log-probs. */
  function lexiconBeam(lp, T, V, lex, opt) {
    opt = opt || {};
    var beamW = opt.beam || 32, lmW = opt.lm_weight == null ? 1.0 : opt.lm_weight;
    var bonus = opt.word_bonus == null ? 2.0 : opt.word_bonus, prune = opt.prune || 12.0, SPACE = 1;
    var beams = [{ words: [], part: [], pb: 0, pnb: NEG, node: lex.root, lm: 0 }];
    for (var t = 0; t < T; t++) {
      var o = t * V, mx = NEG;
      for (var v = 0; v < V; v++) if (lp[o + v] > mx) mx = lp[o + v];
      var cands = [];
      for (v = 1; v < V; v++) if (lp[o + v] > mx - prune) cands.push(v);
      var lpb = lp[o], nxt = new Map();
      var add = function (words, part, pb, pnb, node, lm) {
        var key = words.join(" ") + "|" + part.join(",");
        var s = nxt.get(key);
        if (!s) nxt.set(key, { words: words, part: part, pb: pb, pnb: pnb, node: node, lm: lm });
        else { s.pb = lse(s.pb, pb); s.pnb = lse(s.pnb, pnb); }
      };
      for (var b = 0; b < beams.length; b++) {
        var s = beams[b], tot = lse(s.pb, s.pnb);
        var last = s.part.length ? s.part[s.part.length - 1] : (s.words.length ? SPACE : null);
        add(s.words, s.part, tot + lpb, NEG, s.node, s.lm);
        if (last !== null) add(s.words, s.part, NEG, s.pnb + lp[o + last], s.node, s.lm);
        for (var ci = 0; ci < cands.length; ci++) {
          var c = cands[ci], p = lp[o + c], base = c === last ? s.pb + p : tot + p;
          if (c === SPACE) {
            if (s.node.w === null) continue;
            var w = s.node.w;
            add(s.words.concat([w]), [], NEG, base, lex.root, s.lm + lmW * lex.logp[w] + bonus);
          } else {
            var child = s.node.c.get(c);
            if (child) add(s.words, s.part.concat([c]), NEG, base, child, s.lm);
          }
        }
      }
      beams = Array.from(nxt.values());
      beams.sort(function (x, y) { return (lse(y.pb, y.pnb) + y.lm) - (lse(x.pb, x.pnb) + x.lm); });
      if (beams.length > beamW) beams.length = beamW;
    }
    var best = "", bestScore = NEG;
    beams.forEach(function (s) {
      var score = lse(s.pb, s.pnb) + s.lm, words = s.words;
      if (s.part.length) {
        if (s.node.w === null) return;
        score += lmW * lex.logp[s.node.w] + bonus;
        words = words.concat([s.node.w]);
      }
      if (score > bestScore) { bestScore = score; best = words.join(" "); }
    });
    return best;
  }

  /** Word-level alignment for colouring: returns [{word, ok}] for the hypothesis and the error count. */
  function alignWords(ref, hyp) {
    var r = ref.split(" ").filter(Boolean), h = hyp.split(" ").filter(Boolean);
    var d = [];
    for (var i = 0; i <= r.length; i++) { d.push([i]); }
    for (var j = 1; j <= h.length; j++) d[0][j] = j;
    for (i = 1; i <= r.length; i++)
      for (j = 1; j <= h.length; j++)
        d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] === h[j - 1] ? 0 : 1));
    var ok = h.map(function () { return false; });
    i = r.length; j = h.length;
    while (i > 0 && j > 0) {
      if (r[i - 1] === h[j - 1] && d[i][j] === d[i - 1][j - 1]) { ok[j - 1] = true; i--; j--; }
      else if (d[i][j] === d[i - 1][j - 1] + 1) { i--; j--; }
      else if (d[i][j] === d[i - 1][j] + 1) i--;
      else j--;
    }
    return { words: h.map(function (w, k) { return { word: w, ok: ok[k] }; }), errors: d[r.length][h.length], refWords: r.length };
  }

  var api = { greedy: greedy, buildLexicon: buildLexicon, lexiconBeam: lexiconBeam, alignWords: alignWords };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AfonoDecode = api;
})(typeof window !== "undefined" ? window : this);
