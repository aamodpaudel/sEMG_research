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
    __slots__ = ("children", "word")

    def __init__(self):
        self.children = {}
        self.word = None


class Lexicon:
    def __init__(self, words, counts=None):
        words = sorted({w for w in words if w and all(c in CHAR_TO_ID and c != " " for c in w)})
        self.words = words
        self.root = _Node()
        counts = counts or Counter()
        total = sum(counts[w] + 1 for w in words)
        self.logp = {w: math.log((counts[w] + 1) / total) for w in words}
        for w in words:
            node = self.root
            for c in w:
                node = node.children.setdefault(CHAR_TO_ID[c], _Node())
            node.word = w

    def __len__(self):
        return len(self.words)


def _lse(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def lexicon_beam_search(logprobs, lex, beam=32, lm_weight=1.0, word_bonus=2.0, prune=12.0):
    """logprobs: (T, V) CTC log-probabilities. Returns best word sequence from the lexicon."""
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
                    w = node.word
                    add((words + (w,), ()), NEG_INF, base, lex.root,
                        lm + lm_weight * lex.logp[w] + word_bonus)
                else:
                    child = node.children.get(c)
                    if child is not None:
                        add((words, part + (c,)), NEG_INF, base, child, lm)
        beams = dict(sorted(nxt.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][3],
                            reverse=True)[:beam])
    best, best_score = "", NEG_INF
    for (words, part), (pb, pnb, node, lm) in beams.items():
        score = _lse(pb, pnb) + lm
        if part:
            if node.word is None:
                continue                                                   # unfinished word
            score += lm_weight * lex.logp[node.word] + word_bonus
            words = words + (node.word,)
        if score > best_score:
            best, best_score = " ".join(words), score
    return best
