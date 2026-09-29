"""Experiment registry: one JSON line per training / calibration / evaluation run.

Every run appends to runs/registry.jsonl. `python -m semg.site_data` turns the registry
into the website's progress data, so the site always shows what was actually measured.
"""
import datetime as _dt
import json
import os
import platform
import subprocess

REGISTRY = os.environ.get("AFONO_REGISTRY", "runs/registry.jsonl")


def _now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def log_run(kind: str, run_id: str, **fields) -> dict:
    """kind: 'train' | 'calibrate' | 'evaluate'. Extra fields are stored verbatim."""
    rec = {"time": _now(), "kind": kind, "run_id": run_id, "git": _git_rev(),
           "host": platform.node(), **fields}
    os.makedirs(os.path.dirname(REGISTRY) or ".", exist_ok=True)
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def read_runs(path: str = None) -> list[dict]:
    path = path or REGISTRY
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def headline_metrics(report: dict) -> dict:
    """Flatten an evaluate report into the few numbers we track over time."""
    out = {}
    if "closed_vocab_dates" in report:
        out["dates_word_acc"] = 1 - report["closed_vocab_dates"]["wer"]
    for n, v in report.get("closed_n", {}).items():
        out[f"closed{n}_word_acc"] = v["word_accuracy"]
    if "open_lexicon" in report:
        out["open_wer"] = report["open_lexicon"]["wer"]
    if "open_greedy" in report:
        out["open_greedy_cer"] = report["open_greedy"]["cer"]
    return out
