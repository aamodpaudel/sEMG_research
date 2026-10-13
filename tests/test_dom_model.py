"""Tests for the Afono DOM Model (extension/dom-model.js) and the Tier 1 grammar (extension/commands.js).

Pure-logic tests run the JS in Node; DOM tests run in headless Chrome through semg.dom_eval. Each group is
skipped when its runtime is missing. Nothing here writes to the run registry.
"""
import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXT = os.path.join(ROOT, "extension")
NODE = shutil.which("node")


def js(expr):
    """Evaluate a JS expression with AfonoDOM as D and AfonoCommands as C; return its JSON value."""
    code = (f"const D = require({json.dumps(os.path.join(EXT, 'dom-model.js'))});"
            f"const C = require({json.dumps(os.path.join(EXT, 'commands.js'))});"
            f"process.stdout.write(JSON.stringify({expr}));")
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


need_node = pytest.mark.skipif(NODE is None, reason="node not installed")


@need_node
@pytest.mark.parametrize("name,label", [
    ("Sign in", "sign in"),
    ("Sign in to GitHub", "sign in"),
    ("Log out of your account", "log out"),
    ("Share on Twitter", "share twitter"),
    ("About us", "about"),                      # "us" is a stop word
    ("Your message", "message"),
    ("Add to cart", "add cart"),
    ("Skip to main content", "skip main"),
    ("Star 12.3k", "star"),                     # counts are dropped
    ("157 comments", "comments"),
    ("Page 2", "page two"),                     # small ordinal after a word is spelled out
    ("7Branches", "branches"),
    ("Café Menu", "cafe menu"),
    ("\U0001F4F7 Share this photo!", "share photo"),
    ("json — JSON encoder and decoder", "json encoder"),
    ("One of the five men arrested", "men arrested"),
    ("Click here", "click"),
    ("", ""),
    ("日本語", ""),                  # nothing speakable in a-z
])
def test_short_label(name, label):
    assert js(f"D.shortLabel({json.dumps(name)})") == label


@need_node
def test_labels_use_decoder_charset():
    names = ["Hello, World!", "Don’t miss", "Résumé (PDF)", "C++ & Rust", "100% free", "e-mail"]
    for lab in js(f"{json.dumps(names)}.map(D.shortLabel)"):
        assert all(c.isascii() and (c.isalpha() or c in " '") for c in lab), lab
        assert 0 < len(lab.split()) <= 3


@need_node
def test_spell():
    assert js("[1, 13, 20, 42, 100, 142].map(D.spell)") == [
        "one", "thirteen", "twenty", "forty two", "one hundred", "one hundred forty two"]


@need_node
def test_duplicates_numbered_in_reading_order():
    out = js("D.numberDuplicates([{base:'reply'},{base:'share'},{base:'reply'},{base:'reply two'},{base:'reply'}])"
             ".map(t => [t.label, t.display])")
    labels = [o[0] for o in out]
    assert labels[0] == "reply one" and out[0][1] == "reply 1"
    assert labels[1] == "share"
    assert labels[2] == "reply three"            # "reply two" is a real label elsewhere, so it is skipped
    assert labels[3] == "reply two"
    assert labels[4] == "reply four"
    assert len(set(labels)) == len(labels)


@need_node
def test_lexicon_parts():
    lex = js("D.lexicon([{label:'reply one',kind:'button'},{label:'reply two',kind:'button'},"
             "{label:'search products',kind:'searchbox'}])")
    assert set(lex["labels"]) == {"reply", "one", "two", "search", "products"}
    assert {"click", "scroll", "search", "enter"} <= set(lex["commands"])
    assert set(lex["numbers"]) == {"one", "two", "three"}   # badge numbers 1..3
    assert set(lex["all"]) == set(lex["labels"]) | set(lex["commands"]) | set(lex["numbers"])


@need_node
def test_vocabulary_is_flat_word_list():
    words = js("D.vocabulary([{label:'sign in',kind:'button'}])")
    assert isinstance(words, list) and all(isinstance(w, str) and " " not in w for w in words)


@need_node
def test_command_numbers():
    # digit by digit is the intent grammar's form: "two three" = 23, "2 5" from Whisper = 25
    assert js("['one hundred forty two','twenty one','two three','seven','2 5','25'].map(s => C.number(s.split(' ')))") == [
        142, 21, 23, 7, 25, 25]


@need_node
def test_command_targets_by_label_and_number():
    r = js("(() => { const t=[{n:1,label:'reply one'},{n:2,label:'reply two'},{n:3,label:'share'}];"
           "return [C.parse('click reply two',t).target.n, C.parse('click three',t).target.n,"
           "C.parse('reply two',t).target.n, C.parse('scroll down',t).type]; })()")
    assert r == [2, 3, 2, "scroll"]


# ------------------------------------------------------------------ headless Chrome
def _chrome():
    try:
        from semg.dom_eval import find_browser
        find_browser()
        import websockets  # noqa: F401
        return True
    except Exception:
        return False


need_chrome = pytest.mark.skipif(not _chrome(), reason="Chrome/Edge or websockets not available")


@pytest.fixture(scope="module")
def chrome():
    from semg.dom_eval import Chrome
    ch = Chrome()
    yield ch
    ch.close()


@need_chrome
def test_fixture_regression(chrome):
    from semg.dom_eval import run_fixture
    r = run_fixture(chrome, log=False)
    assert r["precision"] == 1 and r["recall"] == 1, r["misses"]
    assert r["label_accuracy"] == 1 and r["name_accuracy"] == 1, r["misses"]
    assert r["labels_unique"]
    assert r["commands_ok"] == r["commands_total"], r["misses"]


@need_chrome
def test_accessible_name_cases(chrome):
    import os as _os
    from semg.dom_eval import TEST, file_url, load_model
    chrome.navigate(file_url(_os.path.join(TEST, "fixture.html")))
    load_model(chrome)
    got = chrome.eval("""(() => {
      const d = document, N = AfonoDOM.accessibleName;
      const q = s => d.querySelector(s);
      return {
        labelledby: N(q('[aria-labelledby="dl1 dl2"]')),
        hiddenRef: N(q('[aria-labelledby=arch]')),
        wrapEmbedded: N(q('#amt')),
        labelledbyValue: N(q('#qty')),
        svgTitle: N(q('[data-gt=notifications]')),
        imageInput: N(q('input[type=image]')),
        shadow: N(q('#player').shadowRoot.querySelector('button')),
      };
    })()""")
    assert got == {"labelledby": "Download report", "hiddenRef": "Archive mail", "wrapEmbedded": "Pay dollars",
                   "labelledbyValue": "Quantity boxes", "svgTitle": "Notifications", "imageInput": "Go",
                   "shadow": "Play video"}


@need_chrome
def test_dom1_labels_unique_and_gt(chrome):
    from semg.dom_eval import page_files, run_dom1
    if not page_files():
        pytest.skip("no DOM-1 snapshots")
    res, per = run_dom1(chrome, log=False, verbose=False)
    assert res["viewport"]["unique_share"] == 1.0
    assert res["page"]["unique_share"] == 1.0
    if "gt" in res:
        assert res["gt"]["recall"] >= 0.95 and res["gt"]["precision"] >= 0.95


# ------------------------------------------------------------------ WebUI loader stub (synthetic page, no network)
def test_webui_loader(tmp_path):
    import gzip
    import importlib.util
    spec = importlib.util.spec_from_file_location("webui_loader", os.path.join(EXT, "test", "webui", "webui_loader.py"))
    wl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wl)
    key = "default_1280-720"

    def put(part, obj):
        with gzip.open(tmp_path / f"{key}-{part}", "wt", encoding="utf-8") as f:
            json.dump(obj, f)
    put("bb.json.gz", {"1": {"x": 0, "y": 0, "width": 100, "height": 30}, "2": {"x": 0, "y": 40, "width": 80, "height": 20},
                       "3": {"x": 0, "y": 900, "width": 80, "height": 20}, "4": {"x": 0, "y": 0, "width": 0, "height": 0}})
    put("axtree.json.gz", {"nodes": [
        {"backendDOMNodeId": 1, "ignored": False, "role": {"value": "button"}, "name": {"value": "Download"}},
        {"backendDOMNodeId": 2, "ignored": False, "role": {"value": "generic"}, "name": {"value": ""}},
        {"backendDOMNodeId": 3, "ignored": False, "role": {"value": "link"}, "name": {"value": "Blog"}}]})
    put("style.json.gz", {"2": {"cursor": "pointer"}})
    put("viewport.json.gz", {"1": True, "2": True, "3": False})
    els = wl.load_page(str(tmp_path), key)
    assert {e["id"] for e in els} == {"1", "2", "3"}              # zero-size box dropped
    boxes, y = wl.detection_targets(els)
    assert len(boxes) == 2 and sorted(y) == [0, 1]
    div = [e for e in els if e["id"] == "2"][0]
    assert div["cursor"] == "pointer" and not div["clickable"]   # the div-with-handler blind spot of AX labels


@need_node
def test_gaze_commands_double_right_out():
    r = js("(function(){ var t=[{n:25,label:'reply'},{n:3,label:'share'}];"
           "var d=C.parse('double 25',t), rc=C.parse('right click share',t), o=C.parse('out',t), z=C.parse('zoom',t), b=C.parse('right',t);"
           "return [d.type,d.mode,d.target.n, rc.mode,rc.target.n, o.type,o.dir, z.dir, b.type]; })()")
    assert r == ["click", "double", 25, "right", 3, "zoom", -1, 1, "unknown"]
