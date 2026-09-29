"""Score the Afono DOM Model on the DOM-0 fixture (extension/test/fixture.html) in headless Chrome.

    python -m semg.dom_eval

DOM-0 is our own 17-target test page (links, buttons, icon buttons, ARIA roles, script-only
click handlers, form fields, and decoys that are hidden, disabled, covered or off-screen).
It checks the rules; the WebUI dataset benchmark comes next.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

from .registry import log_run

CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "google-chrome", "chromium", "chromium-browser"]


def find_browser():
    for c in CANDIDATES:
        if os.path.exists(c) or shutil.which(c):
            return c
    raise RuntimeError("Chrome or Edge not found")


def run(notes=""):
    page = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extension", "test", "run.html"))
    url = "file:///" + page.replace("\\", "/")
    with tempfile.TemporaryDirectory() as prof:
        out = subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--allow-file-access-from-files",
                              f"--user-data-dir={prof}", "--window-size=1200,1000", "--virtual-time-budget=5000",
                              "--dump-dom", url], capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"RESULT (\{.*?\})</pre>", out, re.S)
    if not m:
        raise RuntimeError("benchmark page did not report a result")
    res = json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&"))
    log_run("evaluate", "dom-0", model="dom", benchmark="DOM-0 fixture", metrics={
        "precision": res["precision"], "recall": res["recall"], "label_accuracy": res["label_accuracy"],
        "commands_ok": res["commands_ok"], "commands_total": res["commands_total"]},
        details=res, notes=notes)
    return res


if __name__ == "__main__":
    r = run("DOM Model v0 (rules) on the DOM-0 fixture")
    print({k: v for k, v in r.items() if k != "false_positives"})
    print("false positives / label misses:", r["false_positives"])
