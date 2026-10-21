import json
import os

import pytest

from semg import study


def test_sus_score_bounds_and_known_value():
    assert study.sus_score([5, 1] * 5) == 100.0
    assert study.sus_score([1, 5] * 5) == 0.0
    assert study.sus_score([3] * 10) == 50.0
    with pytest.raises(ValueError):
        study.sus_score([3] * 9)


def test_next_pid():
    assert study.next_pid({}) == "p01"
    assert study.next_pid({"p01": {}, "p07": {}}) == "p08"


def _write(root, pid, name, rec):
    d = os.path.join(root, pid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(rec, f)


def test_summary_compares_with_same_persons_mouse_run(tmp_path):
    root = str(tmp_path)
    row = lambda i, s, res="done": {"task": i, "seconds": s, "result": res, "errors": 0, "undos": 0}
    _write(root, "p01", "1-mouse.json", {"type": "tasks", "pid": "p01", "mode": "mouse",
                                         "tasks": [row(1, 4.0), row(2, 6.0), row(3, 5.0)]})
    _write(root, "p01", "2-whisper.json", {"type": "tasks", "pid": "p01", "mode": "whisper",
                                           "tasks": [row(1, 8.0), row(2, 12.0), row(3, 30.0, "failed")]})
    _write(root, "p01", "3-survey.json", {"type": "survey", "pid": "p01", "sus": 72.5})
    s = study.summarize(root)
    whisper = next(r for r in s["rows"] if r["mode"] == "whisper")
    assert whisper["success_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert whisper["vs_mouse"] == 2.0            # tasks both finished: 20 s vs 10 s
    assert whisper["sus"] == 72.5
    assert s["median_vs_mouse"] == {"whisper": 2.0}


def test_set_login_for_existing_participant(tmp_path):
    root = str(tmp_path)
    pid = study.register("Ann Example", "ann", "secret1", root=root)
    study._save_people({**study.load_people(root), "p02": {"name": "Bo", "consent": True}}, root)
    study.set_login("p02", "bob", "secret2", root=root)
    assert study.check_login("bob", "secret2", root=root) == "p02"
    assert study.check_login("ann", "secret1", root=root) == pid
    with pytest.raises(ValueError):
        study.set_login("p02", "ann", "secret3", root=root)    # taken by p01
