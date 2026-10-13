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
