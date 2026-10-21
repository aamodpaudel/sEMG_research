import json
import time

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app.server as S
    from semg import registry
    monkeypatch.chdir(tmp_path)                       # data/study is relative: keep it in the temp folder
    monkeypatch.setattr(registry, "REGISTRY", str(tmp_path / "registry.jsonl"))
    S.SESSIONS.clear(); S.RUN.clear(); S.CONTEXT.clear()
    return TestClient(S.app), S, tmp_path


H = {"X-Afono-Client": "test"}


def _register(c, user="sam", pw="secret1"):
    return c.post("/api/study/register", headers=H, json={"full_name": "Sam Tester", "username": user, "password": pw,
                                                          "password2": pw, "consent": True, "keep_voice_clips": False})


def test_register_login_and_privacy(client):
    c, S, tmp = client
    assert c.post("/api/study/register", headers=H, json={"full_name": "A B", "username": "ab", "password": "x"}).status_code == 400
    r = _register(c)
    assert r.status_code == 200 and r.json()["pid"] == "p01" and r.json()["first"] == "Sam"
    assert _register(c).status_code == 400                       # username taken
    assert c.post("/api/study/login", headers=H, json={"username": "sam", "password": "wrong!"}).status_code == 401
    tok = c.post("/api/study/login", headers=H, json={"username": "SAM", "password": "secret1"}).json()["token"]
    me = c.get("/api/study/me", headers={**H, "X-Afono-Token": tok}).json()
    assert me["pid"] == "p01" and me["progress"]["first_mode"] == "mouse"
    assert c.get("/api/study/me", headers=H).status_code == 401
    acc = (tmp / "data" / "study" / "accounts.json").read_text()
    assert "secret1" not in acc and "Sam Tester" not in acc       # only a salted hash, no name


def test_voice_driven_task_run_is_checked_and_logged(client):
    c, S, tmp = client
    tok = _register(c).json()["token"]
    h = {**H, "X-Afono-Token": tok}
    v = c.post("/api/study/run/start", headers=h, json={"mode": "whisper", "device": "test buds", "style": "quiet"}).json()
    assert v["phase"] == "ready" and v["n"] == 7
    assert S._study_words() == ("start",)
    urls = ["https://news.ycombinator.com/item?id=1", "https://news.ycombinator.com/item?id=1",
            "https://news.ycombinator.com/news", "https://news.ycombinator.com/newest",
            "https://news.ycombinator.com/newest", "https://news.ycombinator.com/newsguidelines.html",
            "https://news.ycombinator.com/newest"]
    acts = [[], ["scroll down", "scroll down"], ["back"], [], ["zoom in", "zoom out"], [], ["back", "back"]]
    for i in range(7):
        S._study_voice("start")
        assert S._study_words() == ("done", "skip")
        for a in acts[i]:
            c.post("/api/ran", headers=H, json={"text": a, "action": "x"})
        S.CONTEXT["url"] = urls[i]
        S._study_voice("done" if i != 3 else "skip")
    deadline = time.time() + 5
    while S.RUN.get("phase") != "finished" and time.time() < deadline:
        time.sleep(0.1)
    assert S.RUN["phase"] == "finished"
    rows = S.RUN["rows"]
    assert [r["verified"] for r in rows] == [True, True, True, False, True, True, True]
    assert rows[3]["result"] == "skipped" and rows[6]["undos"] == 1   # one back too many
    log = [json.loads(l) for l in (tmp / "registry.jsonl").read_text().splitlines()]
    assert log[-1]["run_id"] == "demo-1" and log[-1]["participant"] == "p01" and log[-1]["source"] == "nullpoint"
    assert "Sam" not in (tmp / "registry.jsonl").read_text()
    me = c.get("/api/study/me", headers=h).json()
    assert me["progress"]["tasks"] == ["whisper"]


def test_survey_saved_with_score(client):
    c, S, tmp = client
    h = {**H, "X-Afono-Token": _register(c).json()["token"]}
    assert c.post("/api/study/survey", headers=h, json={"answers": [3] * 9, "would_use": 4}).status_code == 400
    r = c.post("/api/study/survey", headers=h, json={"answers": [5, 1] * 5, "would_use": 4, "open": {"best": "fast"}})
    assert r.json()["sus"] == 100.0
    assert c.get("/api/study/me", headers=h).json()["progress"]["survey"] is True
