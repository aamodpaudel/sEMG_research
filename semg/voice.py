"""Whisper mode: quiet or whispered speech from an ordinary microphone (earbuds included), decoded against
the words on screen.

General speech recognition of whispers is unreliable, but choosing 1 of the 20-200 phrases a page allows is a
much easier problem. Whisper (Radford et al., 2022) scores every allowed phrase directly: the audio is encoded
once, and each candidate's token sequence is teacher-forced through the decoder to get log p(phrase | audio).
The free transcription is scored the same way and acts as a garbage model, so speech that matches no
candidate is rejected instead of forced onto the nearest command.

    python -m semg.voice decode clip.wav --labels "sign in" "reply one" "reply two"
    python -m semg.voice selftest          # synthetic speech via Windows SAPI, a pipeline check only

Audio stays on the device. Nothing is stored unless the caller asks for it.
"""
import argparse
import json
import math
import os
import re
import threading
import time
from difflib import SequenceMatcher

import numpy as np
import torch

from .text import normalize_text

MODEL_ID = os.environ.get("AFONO_WHISPER_MODEL", "openai/whisper-base.en")
SR = 16000
MAX_SECONDS = 6.0
FORMS = os.environ.get("AFONO_WHISPER_FORMS", "fast")      # "full" also tries Title Case spellings (slower)

# Tier 1 command phrases that need no page target (see vocab/commands.json and extension/commands.js).
COMMAND_PHRASES = ["scroll up", "scroll down", "page up", "page down", "top", "bottom", "zoom in", "zoom out",
                   "back", "next", "stop", "enter", "undo", "copy", "paste", "delete", "send", "tab", "cancel",
                   "yes", "no", "done", "show", "hide", "sleep", "afono"]

_DIGITS = {w: str(i) for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                             "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                                             "sixteen", "seventeen", "eighteen", "nineteen", "twenty"])}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def number_words(n: int) -> str:
    return normalize_text(str(n))


def digitize(text: str) -> str:
    """'reply two' -> 'reply 2', 'click forty two' -> 'click 42' (Whisper writes numbers as digits)."""
    out, words, i = [], text.split(), 0
    while i < len(words):
        w = words[i]
        if w in _TENS and i + 1 < len(words) and words[i + 1] in _DIGITS and 0 < int(_DIGITS[words[i + 1]]) < 10:
            out.append(str(_TENS[w] + int(_DIGITS[words[i + 1]]))); i += 2; continue
        if w in _TENS:
            out.append(str(_TENS[w]))
        elif w in _DIGITS:
            out.append(_DIGITS[w])
        else:
            out.append(w)
        i += 1
    return " ".join(out)


def candidates(targets=None, commands=COMMAND_PHRASES, max_numbers=60):
    """Allowed phrases for the current page.

    Returns [{"say": spoken form, "send": command text for the extension, "target": badge n or None}].
    A target can be said as its label ("sign in"), as "click <label>", or as "click <badge number>".
    """
    out, seen = [], set()

    def add(say, send, target=None):
        say = normalize_text(say)
        if say and say not in seen:
            seen.add(say)
            out.append({"say": say, "send": send, "target": target})

    for c in commands:
        add(c, c)
    for t in targets or []:
        label, n = normalize_text(t.get("label", "")), t.get("n")
        if label:
            add(label, "click " + label, n)
            add("click " + label, "click " + label, n)
        if n is not None and int(n) <= max_numbers:
            add("click " + number_words(int(n)), "click " + str(int(n)), n)
    return out


def _surface_forms(say: str):
    """Spellings Whisper is likely to produce for a spoken phrase: capitalised, digits for numbers, with or
    without a full stop. The phrase's score is the best of them."""
    forms = {say, digitize(say)}
    cased = {f[:1].upper() + f[1:] for f in forms}
    if FORMS == "full":
        cased |= {f.title() for f in forms}
    return sorted({" " + f + p for f in cased for p in ("", ".")})


class Decoder:
    def __init__(self, model_id=MODEL_ID, device=None):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = WhisperProcessor.from_pretrained(model_id)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.model_id = model_id
        tok = self.processor.tokenizer
        self.tok = tok
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")
        prefix = ["<|startoftranscript|>"]
        if not model_id.endswith(".en"):
            prefix += ["<|en|>", "<|transcribe|>"]
        self.prefix = tok.convert_tokens_to_ids(prefix + ["<|notimestamps|>"])
        self.lock = threading.Lock()

    def _encode(self, audio):
        feats = self.processor.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
        return self.model.model.encoder(feats.to(self.device, self.dtype)).last_hidden_state

    @torch.no_grad()
    def _score(self, enc, texts, batch=256):
        """Sum of log p(tokens + end-of-text | audio) for each text, teacher-forced.

        The prompt tokens and the cross-attention keys/values over the audio are the same for every text, so they
        are computed once and shared across the batch; recomputing the audio projections per text was over 90% of
        the cost (15 s -> well under 1 s per clip on CPU, identical scores)."""
        from transformers.cache_utils import EncoderDecoderCache
        seqs = [self.tok.encode(t, add_special_tokens=False) + [self.eot] for t in texts]
        pre = self.model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([self.prefix], device=self.device),
                         use_cache=True)
        lp0 = torch.log_softmax(pre.logits[0, -1].float(), -1).cpu()        # first token, given the prompt
        legacy = pre.past_key_values.to_legacy_cache()                    # per layer: self k, v, cross k, v
        scores = []
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            B, L = len(chunk), max(len(s) for s in chunk)
            first = torch.tensor([lp0[s[0]] for s in chunk])
            if L == 1:
                scores += first.tolist(); continue
            inp = torch.full((B, L - 1), self.eot, dtype=torch.long)
            tgt = torch.full((B, L - 1), -1, dtype=torch.long)
            for j, s in enumerate(chunk):
                inp[j, :len(s) - 1] = torch.tensor(s[:-1])
                tgt[j, :len(s) - 1] = torch.tensor(s[1:])
            # Filled at batch 1, then widened with expand(): a view, so the audio keys are never copied per text.
            cache = EncoderDecoderCache.from_legacy_cache(legacy)
            for part in (cache.self_attention_cache, cache.cross_attention_cache):
                for layer in part.layers:
                    layer.keys, layer.values = layer.keys.expand(B, -1, -1, -1), layer.values.expand(B, -1, -1, -1)
            out = self.model(encoder_outputs=(enc.expand(B, -1, -1),), decoder_input_ids=inp.to(self.device),
                             past_key_values=cache, use_cache=True)
            lp = torch.log_softmax(out.logits.float(), -1)
            mask = tgt >= 0
            g = lp.gather(-1, tgt.clamp(min=0).to(self.device).unsqueeze(-1)).squeeze(-1).cpu()
            scores += (first + (g * mask).sum(1)).tolist()
        return scores

    @torch.no_grad()
    def transcribe(self, enc):
        """Greedy free transcription (the garbage model); short, since commands are short."""
        ids, feed, past = [], list(self.prefix), None
        for _ in range(24):
            out = self.model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([feed], device=self.device),
                             past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            if nxt == self.eot:
                break
            ids.append(nxt)
            feed = [nxt]
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def decode(self, audio, cands, gaze=None, gaze_weight=1.0, margin=4.0, min_p=0.7):
        """Pick the most likely allowed phrase.

        gaze: {badge n: probability} from the Track Model. It re-weights target phrases softly, mixed with a
        uniform floor so a wrong gaze estimate can lower a phrase but never remove it (hard top-k filtering
        lost accuracy in simulation, METRICS.md "Track Model").
        Accepted when the best phrase has posterior >= min_p and is within `margin` nats of the free
        transcription (the garbage model); otherwise the result is returned but not accepted.
        """
        t0 = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32)[: int(MAX_SECONDS * SR)]
        with self.lock:
            enc = self._encode(audio)
            forms, owner = [], []
            for k, c in enumerate(cands):
                for f in _surface_forms(c["say"]):
                    forms.append(f); owner.append(k)
            s = self._score(enc, forms)
            free = self.transcribe(enc)
            free_score = self._score(enc, [" " + free])[0] if free else -1e9
        best = [-1e9] * len(cands)
        for k, v in zip(owner, s):
            best[k] = max(best[k], v)
        acoustic = list(best)
        if gaze:
            ntgt = max(1, sum(1 for c in cands if c.get("target") is not None))
            for k, c in enumerate(cands):
                if c.get("target") is not None:
                    pg = 0.5 * gaze.get(int(c["target"]), 0.0) + 0.5 / ntgt
                    best[k] += gaze_weight * math.log(pg * ntgt)       # 0 when gaze is uninformative
        m = max(best)
        post = np.exp(np.array(best) - m)
        post /= post.sum()
        order = np.argsort(-post)
        top = [{"say": cands[i]["say"], "send": cands[i]["send"], "target": cands[i]["target"], "p": round(float(post[i]), 4),
                "logp": round(acoustic[i], 2)} for i in order[:5]]
        b = order[0]
        # The free transcription agreeing with the choice, up to homophones and spelling ("reply to" / "reply two",
        # "color" / "colour"), confirms it; the log-probability margin only decides when they disagree.
        f = normalize_text(free)
        same = max(SequenceMatcher(None, f, v).ratio() for v in (cands[b]["say"], normalize_text(digitize(cands[b]["say"])))) >= 0.75
        accepted = bool(post[b] >= min_p and (same or acoustic[b] >= free_score - margin))
        return {"say": cands[b]["say"], "send": cands[b]["send"], "p": round(float(post[b]), 4), "accepted": accepted,
                "free": free, "free_logp": round(free_score, 2), "top": top, "n_candidates": len(cands),
                "gaze_used": bool(gaze), "ms": round(1000 * (time.perf_counter() - t0)), "model": self.model_id,
                "device": self.device.type}


_DECODER = None
_DLOCK = threading.Lock()


def get_decoder():
    global _DECODER
    with _DLOCK:
        if _DECODER is None:
            _DECODER = Decoder(device=os.environ.get("AFONO_WHISPER_DEVICE") or None)
        return _DECODER


def read_wav(path):
    import wave
    with wave.open(path, "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype={2: np.int16, 4: np.int32}[width]).astype(np.float32)
    x = x.reshape(-1, ch).mean(1) / (32768.0 if width == 2 else 2 ** 31)
    return resample(x, sr)


def resample(x, sr):
    if sr == SR:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def _sapi(text, path, rate=0):
    """Synthesise speech with the Windows speech engine (for the pipeline self-test only)."""
    import subprocess
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.Rate = {rate}; $s.SetOutputToWaveFile('{path}'); $s.Speak('{text}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)


def selftest(out_dir, n_pages=3, seed=0):
    """Pipeline check with synthetic (not whispered, not human) speech on page-like candidate sets."""
    import random
    rng = random.Random(seed)
    pages = [["sign in", "search", "reply one", "reply two", "reply three", "share", "menu", "settings", "home",
              "next page", "log out", "help"],
             ["add to cart", "buy now", "reviews", "size", "colour", "wishlist", "shipping", "returns", "account"],
             ["new issue", "pull requests", "code", "actions", "wiki", "security", "insights", "fork", "star",
              "watch", "readme", "releases"]][:n_pages]
    dec = get_decoder()
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for pi, labels in enumerate(pages):
        targets = [{"n": i + 1, "label": l} for i, l in enumerate(labels)]
        cands = candidates(targets)
        says = rng.sample([c["say"] for c in cands], min(12, len(cands)))
        for j, say in enumerate(says):
            p = os.path.join(out_dir, f"p{pi}_{j}.wav")
            _sapi(say, p)
            r = dec.decode(read_wav(p), cands)
            rows.append({"page": pi, "said": say, "got": r["say"], "ok": r["say"] == say, "accepted": r["accepted"],
                         "p": r["p"], "free": r["free"], "ms": r["ms"], "n": len(cands)})
            print(f"{'ok ' if rows[-1]['ok'] else 'ERR'} {say!r:28} -> {r['say']!r:28} p={r['p']:.2f} "
                  f"{'acc' if r['accepted'] else 'rej'} free={r['free']!r} {r['ms']} ms")
    acc = sum(r["ok"] for r in rows) / len(rows)
    print(f"synthetic pipeline check: {sum(r['ok'] for r in rows)}/{len(rows)} = {acc:.1%}, "
          f"median {sorted(r['n'] for r in rows)[len(rows) // 2]} candidates")
    return rows, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("wav")
    d.add_argument("--labels", nargs="*", default=[])
    s = sub.add_parser("selftest")
    s.add_argument("--out", default=os.path.join("data", "voice_selftest"))
    s.add_argument("--log", action="store_true", help="append the result to runs/registry.jsonl")
    a = ap.parse_args()
    if a.cmd == "decode":
        cands = candidates([{"n": i + 1, "label": l} for i, l in enumerate(a.labels)])
        print(json.dumps(get_decoder().decode(read_wav(a.wav), cands), indent=1))
    else:
        rows, acc = selftest(a.out)
        if a.log:
            from .registry import log_run
            log_run("evaluate", "voice-selftest", model="voice", benchmark="synthetic-sapi",
                    metrics={"accuracy": acc, "n": len(rows)}, whisper_model=MODEL_ID,
                    notes="Whisper mode pipeline check on Windows SAPI synthetic speech (normal voice, not whispered, "
                          "not a person). Not evidence of accuracy on real whispers.")


if __name__ == "__main__":
    main()
