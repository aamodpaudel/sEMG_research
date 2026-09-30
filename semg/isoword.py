"""Isolated-word proxy (protocol v3): per-word N-way classification inside test sentences.

1. CTC forced alignment of the reference transcript to the model's own log-probabilities gives
   a frame span for every word (oracle segmentation: it uses the reference).
2. The span, padded by 2 frames (16 ms) each side, is scored against N candidates (the true word
   plus N-1 distractors, easy or hard as in semg.page) by CTC log-likelihood.
3. Correct if the true word scores highest.

What it measures: how well the model tells one word from N candidates when word boundaries are
known. It is optimistic compared with real isolated-word use (the boundaries come from the true
word, and the encoder sees the sentence context), and it is not a recording of isolated words.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as taf

from .metrics import _ratio_ci
from .page import word_candidates

PAD = 2


def word_spans(lp, token_seq, space):
    """Frame spans (start, end_exclusive) of each word, or None if alignment fails."""
    T = lp.shape[0]
    if len(token_seq) == 0 or T < len(token_seq):
        return None
    try:
        ali, scores = taf.forced_align(torch.from_numpy(lp)[None].float(),
                                       torch.tensor(token_seq, dtype=torch.int32)[None], blank=0)
    except Exception:
        return None
    spans = taf.merge_tokens(ali[0], scores[0].exp())
    if len(spans) != len(token_seq):
        return None
    words, cur = [], None
    for tok, sp in zip(token_seq, spans):
        if tok == space:
            if cur:
                words.append(cur)
            cur = None
        else:
            cur = (sp.start, sp.end) if cur is None else (cur[0], sp.end)
    if cur:
        words.append(cur)
    return words


def ctc_scores(seg, cands, spell):
    """Log-likelihood of each candidate spelling on a (T, V) log-prob segment."""
    T = seg.shape[0]
    tg = [spell(c) for c in cands]
    x = torch.from_numpy(seg).float()[:, None].expand(T, len(cands), seg.shape[1])
    targets = torch.cat([torch.tensor(t) for t in tg])
    loss = F.ctc_loss(x, targets, torch.full((len(cands),), T, dtype=torch.long),
                      torch.tensor([len(t) for t in tg]), blank=0, reduction="none", zero_infinity=False)
    s = -loss.numpy()
    s[~np.isfinite(s)] = -np.inf
    return s


def isolated_word_accuracy(utts, lps, spell, space, n, variant, pool, index=None):
    """Returns summary dict with accuracy, 95% bootstrap CI over utterances, and counts."""
    correct, total, failed = [], [], 0
    for u, lp in zip(utts, lps):
        ws = u.text.split()
        seq = []
        for i, w in enumerate(ws):
            if i:
                seq.append(space)
            seq.extend(spell(w))
        spans = word_spans(lp, seq, space)
        if spans is None or len(spans) != len(ws):
            failed += 1              # counted as errors, never dropped
            correct.append(0)
            total.append(len(ws))
            continue
        c = 0
        for i, (w, (s, e)) in enumerate(zip(ws, spans)):
            seg = lp[max(0, s - PAD): min(lp.shape[0], e + PAD)]
            cands = word_candidates(f"{u.uid}:{i}", w, n, pool, variant, index)
            sc = ctc_scores(seg, cands, spell)
            c += int(np.argmax(sc) == 0 and np.isfinite(sc[0]))
        correct.append(c)
        total.append(len(ws))
    correct, total = np.array(correct, float), np.array(total, float)
    lo, hi = _ratio_ci(correct, total)
    return {"accuracy": float(correct.sum() / max(1, total.sum())), "accuracy_ci95": [lo, hi],
            "n_words": int(total.sum()), "n_utterances": len(utts), "alignment_failures": failed,
            "n_candidates": n, "variant": variant}
