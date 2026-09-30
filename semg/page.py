"""Page-vocabulary tasks (protocol v3): "the page tells the decoder which ~N words are possible".

For one utterance, the lexicon is the utterance's distinct words plus distractor words from the
training vocabulary, so that it holds N words. When the utterance already has k >= N - ceil(N/3)
distinct words, it still gets ceil(N/3) distractors (so the lexicon can exceed N; every
utterance is kept and the mean lexicon size is reported).

Distractors are deterministic:
- easy: training words ranked by a seeded SHA-256 hash of (utterance id, word)
- hard: for each utterance word in order of first appearance, round-robin, the next-nearest
        training word by character edit distance (ties broken by the same hash)

See METRICS.md, "Page-vocabulary tasks". Single-speaker lab-hardware simulation, not a
measurement on the target hardware or the target user.
"""
import hashlib
import math

try:
    from rapidfuzz.distance import Levenshtein as _lev
    from rapidfuzz import process as _process
except Exception:  # pragma: no cover
    _lev = _process = None

from .text import edit_distance

SEED = "afono-page-v3"


def _h(*parts):
    return int(hashlib.sha256("|".join((SEED,) + parts).encode()).hexdigest()[:12], 16)


def n_distractors(k, n):
    return max(n - k, math.ceil(n / 3))


def easy_distractors(uid, words, pool, d):
    ws = set(words)
    ranked = sorted((w for w in pool if w not in ws), key=lambda w: _h(uid, w))
    return ranked[:d]


class NearestIndex:
    """Nearest training words by character edit distance, cached per query word."""

    def __init__(self, pool, depth=40):
        self.pool = sorted(pool)
        self.depth = depth
        self._cache = {}

    def nearest(self, w):
        if w not in self._cache:
            if _process is not None:
                d = _process.cdist([w], self.pool, scorer=_lev.distance, workers=1)[0]
                pairs = sorted(((int(d[i]), _h("near", w, c), c) for i, c in enumerate(self.pool) if c != w))
            else:
                pairs = sorted(((edit_distance(w, c), _h("near", w, c), c) for c in self.pool if c != w))
            self._cache[w] = [c for _, _, c in pairs[: self.depth + 64]]
        return self._cache[w]


def hard_distractors(uid, words, index, d):
    order = list(dict.fromkeys(words))
    ws, chosen = set(order), []
    taken = set()
    ptr = {w: 0 for w in order}
    lists = {w: index.nearest(w) for w in order}
    while len(chosen) < d:
        progress = False
        for w in order:
            lst = lists[w]
            while ptr[w] < len(lst) and (lst[ptr[w]] in ws or lst[ptr[w]] in taken):
                ptr[w] += 1
            if ptr[w] < len(lst):
                c = lst[ptr[w]]
                chosen.append(c)
                taken.add(c)
                ptr[w] += 1
                progress = True
                if len(chosen) >= d:
                    break
        if not progress:
            break
    return chosen


def page_lexicon(uid, text, n, pool, variant, index=None):
    """Word list for one utterance: its distinct words + distractors (see module docstring)."""
    words = list(dict.fromkeys(text.split()))
    d = n_distractors(len(words), n)
    if variant == "easy":
        extra = easy_distractors(uid, words, pool, d)
    elif variant == "hard":
        extra = hard_distractors(uid, words, index, d)
    else:
        raise ValueError(variant)
    return words + extra


def word_candidates(key, word, n, pool, variant, index=None):
    """Isolated-word proxy: the true word plus n-1 distractors (easy = hashed, hard = nearest)."""
    if variant == "easy":
        return [word] + easy_distractors(key, [word], pool, n - 1)
    return [word] + hard_distractors(key, [word], index, n - 1)
