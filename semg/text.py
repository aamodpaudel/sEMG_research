"""Text normalisation, character vocabulary and error-rate metrics."""
import re
import unicodedata

BLANK = 0
CHARS = ["<blank>", " "] + list("abcdefghijklmnopqrstuvwxyz'")
SPACE = 1
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
        "nine": "ninth", "twelve": "twelfth"}


def _int_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _int_words(rest))
    if 1100 <= n < 2100 and n % 100 != 0 and n // 100 != 20:  # years like 1984, 2019 -> keep simple
        return _int_words(n // 100) + " " + _int_words(n % 100)
    rest = n % 1000
    return _int_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + _int_words(rest))


def _ordinal(words: str) -> str:
    parts = words.split()
    last = parts[-1]
    if last in _ORD:
        last = _ORD[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last = last + "th"
    return " ".join(parts[:-1] + [last])


def _replace_number(m: re.Match) -> str:
    num, suffix = m.group(1), m.group(2)
    words = _int_words(int(num)) if len(num) <= 4 else " ".join(_ONES[int(d)] for d in num)
    return " " + (_ordinal(words) if suffix else words) + " "


def _replace_time(m: re.Match) -> str:
    h, mnt = int(m.group(1)), int(m.group(2))
    words = _int_words(h)
    if mnt:
        words += (" oh " if mnt < 10 else " ") + _int_words(mnt)
    return " " + words + " "


def normalize_text(text: str) -> str:
    """Lowercase, ASCII-fold, spell out numbers, keep only a-z, apostrophe and single spaces."""
    text = re.sub("[\\u2012-\\u2015\\u2026]", " ", text)  # dashes and ellipses separate words (protocol v3)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _replace_time, text)
    text = re.sub(r"(\d+)(st|nd|rd|th)?\b", _replace_number, text)
    text = re.sub(r"[-/:]", " ", text)
    text = re.sub(r"[^a-z' ]", "", text)
    text = re.sub(r"(^|\s)'+|'+(\s|$)", " ", text)  # quotes, not contractions
    return re.sub(r"\s+", " ", text).strip()


def encode(text: str) -> list[int]:
    return [CHAR_TO_ID[c] for c in text]


def decode_ids(ids) -> str:
    return "".join(CHARS[i] for i in ids if i != BLANK)


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    errors = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    return errors / max(1, sum(len(r.split()) for r in refs))


def cer(refs: list[str], hyps: list[str]) -> float:
    errors = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    return errors / max(1, sum(len(r) for r in refs))
