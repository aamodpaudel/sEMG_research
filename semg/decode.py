"""CTC decoding: greedy, and lexicon-constrained prefix beam search with a word unigram prior.

The lexicon decoder can only output words from a given vocabulary. That is how a
"500-word" (or "the 14 buttons on this page") interface gets high accuracy.
"""
import math
from collections import Counter

import numpy as np

from .text import BLANK, CHARS, CHAR_TO_ID, SPACE

NEG_INF = -float("inf")


def greedy(logprobs: np.ndarray) -> str:
    ids = logprobs.argmax(-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK:
            out.append(CHARS[i])
        prev = i
    return " ".join("".join(out).split())


class _Node:
    __slots__ = ("children", "word", "words")

    def __init__(self):
        self.children = {}
        self.word = None          # first word ending here
        self.words = None         # all words ending here (homophones share a node in a phone lexicon)


class Lexicon:
    """Allowed words (a trie over characters) plus a word prior.

    With `bigrams` (Counter of (prev, word) pairs, "<s>" = sentence start) the prior is an
    interpolated absolute-discount bigram model; with `lm` (a semg.lm.NGramLM) it is that
    model renormalised over these words; otherwise it is an add-one unigram (uniform when
    no counts are given).
    `spell` maps a word to CTC token ids (default: characters); `space` is the word-boundary id.
    """

    def __init__(self, words, counts=None, bigrams=None, discount=0.75, lm=None, spell=None, space=SPACE):
        if spell is None:
            words = sorted({w for w in words if w and all(c in CHAR_TO_ID and c != " " for c in w)})
            spell = lambda w: [CHAR_TO_ID[c] for c in w]  # noqa: E731
        else:
            words = sorted({w for w in words if w})
        self.words, self.space = words, space
        self.lm, self.vocab = lm, frozenset(words)
        self.root = _Node()
        counts = counts or Counter()
        total = sum(counts[w] + 1 for w in words)
        self.logp = {w: math.log((counts[w] + 1) / total) for w in words}
        self.uni = {w: (counts[w] + 1) / total for w in words}
        self.bi, self.ctx, self.types, self.d = None, {}, {}, discount
        if bigrams:
            ws = set(words)
            self.bi = Counter({k: c for k, c in bigrams.items() if k[1] in ws})
            for (v, w), c in self.bi.items():
                self.ctx[v] = self.ctx.get(v, 0) + c
                self.types[v] = self.types.get(v, 0) + 1
        self._cache = {}
        for w in words:
            node = self.root
            for c in spell(w):
                node = node.children.setdefault(c, _Node())
            if node.word is None:
                node.word, node.words = w, [w]
            else:
                node.words.append(w)

    def __len__(self):
        return len(self.words)

    def score(self, prev, w):
        """log P(w | prev); `prev` is the previous word or a tuple of previous words."""
        if self.lm is not None:
            hist = prev if isinstance(prev, tuple) else ((prev,) if prev != "<s>" else ())
            key = (hist[-(self.lm.order - 1):] if self.lm.order > 1 else (), w)
            if key not in self._cache:
                self._cache[key] = self.lm.logprob(key[0], w, self.vocab)
            return self._cache[key]
        if isinstance(prev, tuple):
            prev = prev[-1] if prev else "<s>"
        if self.bi is None:
            return self.logp[w]
        key = (prev, w)
        if key not in self._cache:
            n = self.ctx.get(prev, 0)
            if n == 0:
                p = self.uni[w]
            else:
                p = max(self.bi.get(key, 0) - self.d, 0) / n + self.d * self.types[prev] / n * self.uni[w]
            self._cache[key] = math.log(p)
        return self._cache[key]


def bigram_counts(texts):
    """Counter of (previous word, word) pairs, with "<s>" before each sentence."""
    c = Counter()
    for t in texts:
        ws = ["<s>"] + t.split()
        c.update(zip(ws, ws[1:]))
    return c


def _lse(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def lexicon_beam_search(logprobs, lex, beam=32, lm_weight=1.0, word_bonus=2.0, prune=12.0, space_optional=False):
    """logprobs: (T, V) CTC log-probabilities. Returns best word sequence from the lexicon.
    space_optional: a word may also end without an emitted word-boundary token (the next word starts
    directly), for models that rarely emit the boundary in silent speech."""
    SPACE = lex.space
    # state key: (completed words, partial word chars) -> [p_blank, p_nonblank, trie node, lm score]
    beams = {((), ()): [0.0, NEG_INF, lex.root, 0.0]}
    for t in range(logprobs.shape[0]):
        lp = logprobs[t]
        lp_blank = float(lp[BLANK])
        cands = [int(c) for c in np.nonzero(lp > lp.max() - prune)[0] if c != BLANK]
        nxt = {}

        def add(key, pb, pnb, node, lm):
            s = nxt.get(key)
            if s is None:
                nxt[key] = [pb, pnb, node, lm]
            else:
                s[0], s[1] = _lse(s[0], pb), _lse(s[1], pnb)

        for (words, part), (pb, pnb, node, lm) in beams.items():
            tot = _lse(pb, pnb)
            last = part[-1] if part else (SPACE if words else None)
            add((words, part), tot + lp_blank, NEG_INF, node, lm)          # emit blank
            if last is not None:                                           # repeat last symbol
                add((words, part), NEG_INF, pnb + float(lp[last]), node, lm)
            for c in cands:
                p = float(lp[c])
                base = pb + p if c == last else tot + p                    # repeats need a blank between
                if c == SPACE:
                    if node.word is None:                                  # only end a complete word
                        continue
                    for w in node.words:
                        add((words + (w,), ()), NEG_INF, base, lex.root,
                            lm + lm_weight * lex.score(words, w) + word_bonus)
                else:
                    child = node.children.get(c)
                    if child is not None:
                        add((words, part + (c,)), NEG_INF, base, child, lm)
                    if space_optional and node.word is not None:
                        nchild = lex.root.children.get(c)
                        if nchild is not None:
                            for w in node.words:
                                nw = words + (w,)
                                add((nw, (c,)), NEG_INF, base, nchild,
                                    lm + lm_weight * lex.score(words, w) + word_bonus)
        beams = dict(sorted(nxt.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][3],
                            reverse=True)[:beam])
    best, best_score = "", NEG_INF
    for (words, part), (pb, pnb, node, lm) in beams.items():
        score = _lse(pb, pnb) + lm
        if part:
            if node.word is None:
                continue                                                   # unfinished word
            ends = [(score + lm_weight * lex.score(words, w) + word_bonus, words + (w,)) for w in node.words]
        else:
            ends = [(score, words)]
        for sc, ws in ends:
            if sc > best_score:
                best, best_score = " ".join(ws), sc
    return best
