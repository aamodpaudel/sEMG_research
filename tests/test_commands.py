"""Tier 1 command pipeline on synthetic 4-channel recordings (no dataset or hardware needed)."""
import json
import types

import numpy as np

from semg import commands


def test_starter_set_is_20_to_30_items():
    items = commands.starter_items()
    assert 20 <= len(items) <= 30 and len(set(items)) == len(items)
    groups = json.load(open(commands.COMMANDS_FILE))["groups"]
    allowed = {p for g in groups.values() for p in g}
    assert set(items) <= allowed


def test_session_order_is_deterministic_and_complete():
    a = commands.session_order(["up", "down"], 3, 1)
    assert a == commands.session_order(["up", "down"], 3, 1) and len(a) == 6
    assert a != commands.session_order(["up", "down"], 3, 2) or len(set(a)) == 6


def test_record_synth_train_eval(tmp_path, monkeypatch):
    monkeypatch.setenv("AFONO_USER_ROOT", str(tmp_path / "users"))
    monkeypatch.setattr(commands, "log_run", lambda *a, **k: None)
    items = ["up", "down", "left", "right", "stop", "yes"]
    ns = types.SimpleNamespace
    commands.cmd_synth(ns(user="t", items=items, sessions=3, reps=8, channels=4, seed=0))
    # simulated serial recording path writes prompted, session-tagged recordings too
    commands.cmd_record(ns(user="rec", items=["up", "down"], session=1, reps=2, seed=0, channels=4, simulate=True, port=None))
    p = commands.Profile("rec")
    assert len(commands.dataset(p)) == 4 and all(r[2] == 1 for r in commands.dataset(p))
    res = commands.cmd_eval(ns(user="t", items=items, base="none", epochs=80, out=str(tmp_path / "run"), notes=""))
    assert res["loso"]["n"] == 6 * 8 * 3
    assert res["loso"]["accuracy"] > 0.5          # chance is 1/6
    assert (tmp_path / "run" / "report.json").exists()
