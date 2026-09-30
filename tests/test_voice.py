"""Whisper mode: candidate phrases and number spelling (no speech model needed)."""
from semg import voice as V


def test_digitize():
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
