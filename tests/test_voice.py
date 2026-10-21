"""Whisper mode: candidate phrases and number spelling (no speech model needed)."""
from semg import voice as V


def test_digitize():
    assert V.digitize("click two five") == "click 25"           # intent grammar: digit by digit
    assert V.digitize("click one two three four") == "click 123 4"
    assert V.spoken_digits(25) == "two five"
    assert V.digitize("reply two") == "reply 2"
    assert V.digitize("click forty two") == "click 42"
    assert V.digitize("click twenty") == "click 20"
    assert V.digitize("sign in") == "sign in"


def test_candidates_cover_labels_clicks_numbers_and_commands():
    c = V.candidates([{"n": 1, "label": "sign in"}, {"n": 12, "label": "reply two"}])
    say = {x["say"]: x for x in c}
    assert say["sign in"]["send"] == "click sign in" and say["sign in"]["target"] == 1
    assert say["click reply two"]["target"] == 12
    assert say["click twelve"]["send"] == "click 12"
    assert "scroll down" in say and say["scroll down"]["target"] is None
    assert len(say) == len(c)                                   # no duplicate phrases


def test_surface_forms_include_digits():
    f = V._surface_forms("click reply two")
    assert " Click reply 2" in f and " Click reply two." in f


def test_decode_decides_on_actions_not_phrases():
    """'click twenty five' and 'click two five' share the spelling 'Click 25'; together they must be accepted."""
    import threading
    import types
    d = object.__new__(V.Decoder)
    d.lock, d.model_id, d.device = threading.Lock(), "stub", types.SimpleNamespace(type="cpu")
    d._encode = lambda audio: None
    d.transcribe = lambda enc: "Click 25."
    d._score = lambda enc, texts: [(-1.0 if t.strip().rstrip(".") == "Click 25" else -30.0) for t in texts]
    cands = V.candidates([{"n": 25, "label": "reply"}, {"n": 3, "label": "share"}])
    r = d.decode([0.0] * 16000, cands)
    assert r["send"] == "click 25" and r["accepted"] and r["p"] > 0.99


def test_selftest_prompts_weighted_like_real_use():
    labels = ["new", "past", "comments", "guidelines", "php eight two", "upvote", "upvote two", "hide three",
              "duckdb 2.0.0 lts", "a very long story title that nobody would say"]
    targets = [{"n": i + 1, "label": l} for i, l in enumerate(labels)] + [{"n": 25, "label": "upvote nine"}]
    ps = V.selftest_prompts(targets, n=20, seed=1)
    allowed = {c["say"] for c in V.candidates(targets)}
    assert len(ps) == 20 and all(p in allowed for p in ps)
    nums = [p for p in ps if p.startswith("click ") and p.split()[1] in ("one", "two", "three", "four", "five",
                                                                         "six", "seven", "eight", "nine", "ten")]
    assert len(nums) >= 9                                   # about half are "click" + a number
    assert sum(p in V.PROMPT_COMMANDS for p in ps) >= 5
    for bad in ("upvote two", "hide three", "duckdb 2.0.0 lts", "upvote nine"):
        assert bad not in ps
    assert all(len(p.split()) <= 4 for p in ps)
