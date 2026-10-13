"""Gaze x sEMG soft fusion: semg.page.gaze_word_weights (used by Nullpoint's page vocabulary)."""
import pytest

from semg.page import gaze_word_weights

TARGETS = [{"n": 1, "label": "Sign in"}, {"n": 2, "label": "Register"}, {"n": 3, "label": "Help"},
           {"n": 4, "label": "Reply one"}, {"n": 5, "label": "Reply two"}]


def test_no_gaze_or_no_targets_changes_nothing():
    assert gaze_word_weights(TARGETS, {}) == {}
    assert gaze_word_weights([], {1: 1.0}) == {}
    assert gaze_word_weights(TARGETS, {9: 1.0}) == {}         # gaze on nothing the page listed


def test_uniform_gaze_is_neutral():
    w = gaze_word_weights(TARGETS, {t["n"]: 0.2 for t in TARGETS})
    assert all(v == pytest.approx(1.0) for v in w.values())


def test_gaze_boosts_softly_and_never_removes():
    w = gaze_word_weights(TARGETS, {2: 1.0})
    n = len(TARGETS)
    assert w["register"] == pytest.approx(0.5 * n + 0.5)        # all the gaze on one target
    assert w["help"] == pytest.approx(0.5)                      # floor: lowered, never zero
    assert "scroll" not in w                                    # command words keep weight 1 (absent = 1)
    # a word shared by several targets averages them: looking at "reply two" lifts "reply" less than "two"
    w = gaze_word_weights(TARGETS, {5: 1.0})
    assert w["two"] > w["reply"] > w["one"]
