"""Word n-gram language model: interpolated modified Kneser-Ney (Chen & Goodman 1999).

Trained on training transcripts only (no external corpus). Lower orders use continuation
counts; the lowest order is interpolated with a uniform distribution over the decoder's
lexicon, so every lexicon word (including ones never seen in training) has non-zero mass.

`NGramLM.logprob(hist, w, vocab)` returns log P(w | hist) renormalised over `vocab`
(a frozenset of allowed words), which is what a closed-vocabulary decoder needs.
"""
import math
from collections import Counter, defaultdict

BOS = "<s>"


class NGramLM:
    def __init__(self, texts, order=3):
        self.order = order
        counts = [None] + [Counter() for _ in range(order)]    # counts[n][ngram tuple]
        for t in texts:
            ws = [BOS] * (order - 1) + t.split()
            for i in range(order - 1, len(ws)):
                for n in range(1, order + 1):
                    if i - n + 1 < 0:
                        break
                    counts[n][tuple(ws[i - n + 1:i + 1])] += 1
        # continuation counts for lower orders: number of distinct left extensions
        self.c = [None] * (order + 1)
        self.c[order] = counts[order]
        for n in range(order - 1, 0, -1):
            cc = Counter()
            for g in counts[n + 1]:
                cc[g[1:]] += 1
            # n-grams that start with <s> have no real left extension: keep their raw counts
            for g, v in counts[n].items():
                if g[0] == BOS:
                    cc[g] = v
            self.c[n] = cc
        self.D = [None] * (order + 1)
        for n in range(1, order + 1):
            coc = Counter(v for v in self.c[n].values() if v <= 4)
            n1, n2, n3, n4 = (coc.get(k, 0) for k in (1, 2, 3, 4))
            if min(n1, n2, n3, n4) == 0:
                self.D[n] = (0.5, 1.0, 1.5)
                continue
            y = n1 / (n1 + 2 * n2)
            self.D[n] = (max(0.05, 1 - 2 * y * n2 / n1), max(0.05, 2 - 3 * y * n3 / n2), max(0.05, 3 - 4 * y * n4 / n3))
        # per-context totals, backoff weights and seen continuations
        self.tot, self.gamma, self.cont = [None] * (order + 1), [None] * (order + 1), [None] * (order + 1)
        for n in range(1, order + 1):
            tot, nk, cont = Counter(), defaultdict(lambda: [0, 0, 0]), defaultdict(dict)
            for g, v in self.c[n].items():
                h = g[:-1]
                tot[h] += v
                nk[h][min(v, 3) - 1] += 1
                cont[h][g[-1]] = v
            d = self.D[n]
            self.tot[n] = tot
            self.gamma[n] = {h: (d[0] * k[0] + d[1] * k[1] + d[2] * k[2]) / tot[h] for h, k in nk.items()}
            self.cont[n] = cont
        self._p, self._z = {}, {}

    def _disc(self, n, v):
        return max(0.0, v - self.D[n][min(v, 3) - 1])

    def _prob(self, hist, w, vsize):
        """Unnormalised-over-subset P(w | hist), hist a tuple of length order-1 (or shorter)."""
        key = (hist, w, vsize)
        if key in self._p:
            return self._p[key]
        n = len(hist) + 1
        lower = (1.0 / vsize) if n == 1 else self._prob(hist[1:], w, vsize)
        if n == 1:
            tot = self.tot[1].get((), 0)
            p = (self._disc(1, self.c[1].get((w,), 0)) / tot + self.gamma[1][()] * lower) if tot else lower
        else:
            tot = self.tot[n].get(hist, 0)
            if tot == 0:
                p = lower
            else:
                p = self._disc(n, self.c[n].get(hist + (w,), 0)) / tot + self.gamma[n][hist] * lower
        self._p[key] = p
        return p

    def _norm(self, hist, vocab):
        """Sum over w in vocab of _prob(hist, w), computed through the seen continuations."""
        key = (hist, id(vocab))
        if key in self._z:
            return self._z[key]
        n = len(hist) + 1
        vsize = len(vocab)
        if n == 1:
            tot = self.tot[1].get((), 0)
            if not tot:
                z = 1.0
            else:
                seen = sum(self._disc(1, v) for w, v in self.cont[1].get((), {}).items() if w in vocab)
                z = seen / tot + self.gamma[1][()]      # uniform part sums to 1 over vocab
        else:
            lz = self._norm(hist[1:], vocab)
            tot = self.tot[n].get(hist, 0)
            if tot == 0:
                z = lz
            else:
                seen = sum(self._disc(n, v) for w, v in self.cont[n][hist].items() if w in vocab)
                z = seen / tot + self.gamma[n][hist] * lz
        self._z[key] = z
        return z

    def logprob(self, hist, w, vocab):
        """log P(w | hist) over `vocab`; hist = previous words (most recent last), BOS-padded here."""
        k = self.order - 1
        hist = tuple(hist[-k:]) if k else ()
        if len(hist) < k:
            hist = (BOS,) * (k - len(hist)) + hist
        return math.log(self._prob(hist, w, len(vocab)) / self._norm(hist, vocab))

    def perplexity(self, texts, vocab):
        lp, n = 0.0, 0
        for t in texts:
            ws = t.split()
            for i, w in enumerate(ws):
                if w in vocab:
                    lp += self.logprob(tuple(ws[:i]), w, vocab)
                    n += 1
        return math.exp(-lp / max(1, n))
