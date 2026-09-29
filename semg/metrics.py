"""Research metrics: error rates with bootstrap confidence intervals, paired comparisons,
information transfer rate (ITR) and speed.

Conventions (see METRICS.md):
- WER/CER are corpus-level: total edit operations / total reference words (characters).
- Word accuracy = 1 - WER (can be negative for WER > 1; we clip at 0 only for ITR).
- CIs: percentile bootstrap over utterances, 2,000 resamples, fixed seed.
- ITR: Wolpaw et al. (2002) bits per selection, one selection = one word from an N-word vocabulary.
"""
import math

import numpy as np

from .text import edit_distance

N_BOOT = 2000
SEED = 1234


def utterance_errors(refs, hyps):
    """Per-utterance (word_errors, ref_words, char_errors, ref_chars)."""
    rows = []
    for r, h in zip(refs, hyps):
        rows.append((edit_distance(r.split(), h.split()), len(r.split()), edit_distance(r, h), len(r)))
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def _ratio_ci(num, den, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = num[idx].sum(1) / np.maximum(1, den[idx].sum(1))
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def wolpaw_bits(n_choices, p):
    """Bits per selection for an N-way choice with accuracy p (Wolpaw 2002)."""
    if n_choices < 2:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    if p <= 1.0 / n_choices:
        return 0.0
    b = math.log2(n_choices) + p * math.log2(p)
    if p < 1:
        b += (1 - p) * math.log2((1 - p) / (n_choices - 1))
    return b


def score(refs, hyps, seconds=None, vocab_size=None, compute_ms=None):
    """Full research summary for one task."""
    e = utterance_errors(refs, hyps)
    words, chars = e[:, 1].sum(), e[:, 3].sum()
    w, c = e[:, 0].sum() / max(1, words), e[:, 2].sum() / max(1, chars)
    w_lo, w_hi = _ratio_ci(e[:, 0], e[:, 1])
    c_lo, c_hi = _ratio_ci(e[:, 2], e[:, 3])
    out = {"n_utterances": len(refs), "n_words": int(words), "wer": w, "wer_ci95": [w_lo, w_hi],
           "cer": c, "cer_ci95": [c_lo, c_hi], "word_accuracy": 1 - w, "word_accuracy_ci95": [1 - w_hi, 1 - w_lo],
           "sentence_accuracy": float(np.mean([r == h for r, h in zip(refs, hyps)]))}
    if seconds is not None:
        total_s = float(np.sum(seconds))
        wpm = 60.0 * words / max(1e-9, total_s)
        out["speaking_wpm"] = wpm
        out["correct_wpm"] = wpm * max(0.0, 1 - w)
        if vocab_size:
            bits = wolpaw_bits(vocab_size, 1 - w)
            out["vocab_size"] = int(vocab_size)
            out["bits_per_word"] = bits
            out["itr_bits_per_min"] = bits * wpm
        if compute_ms is not None:
            out["real_time_factor"] = float(np.sum(compute_ms)) / 1000.0 / max(1e-9, total_s)
    return out


def paired_bootstrap(refs, hyps_a, hyps_b, n_boot=N_BOOT, seed=SEED):
    """WER(B) - WER(A) on the same utterances, with 95% CI and one-sided p(B not better than A)."""
    ea, eb = utterance_errors(refs, hyps_a), utterance_errors(refs, hyps_b)
    words = ea[:, 1]
    rng = np.random.default_rng(seed)
    n = len(refs)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = (eb[idx, 0].sum(1) - ea[idx, 0].sum(1)) / np.maximum(1, words[idx].sum(1))
    delta = (eb[:, 0].sum() - ea[:, 0].sum()) / max(1, words.sum())
    return {"n_utterances": n, "wer_a": float(ea[:, 0].sum() / max(1, words.sum())),
            "wer_b": float(eb[:, 0].sum() / max(1, words.sum())), "delta_wer": float(delta),
            "delta_ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_b_not_better": float(np.mean(d >= 0))}
