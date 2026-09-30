"""Evaluate the Afono DOM Model (extension/dom-model.js) in headless Chrome.

    python -m semg.dom_eval                 # DOM-0 fixture + DOM-1 real-page benchmark, logs both to the registry
    python -m semg.dom_eval fixture         # DOM-0 fixture only
    python -m semg.dom_eval dom1            # DOM-1 only
    python -m semg.dom_eval snapshot [name] # (re)capture real pages into extension/test/pages/ (needs network)
    python -m semg.dom_eval candidates NAME # screenshot + candidate list for hand annotation (scratch output)

DOM-0 (extension/test/fixture.html) is our own test page: links, buttons, icon buttons, ARIA naming (labelledby,
label-for, wrapping labels, title, svg title), duplicates, nesting, shadow DOM, custom checkboxes and decoys
(hidden, disabled, covered, transparent, inert, off-screen, focusable scroll region). We wrote it, so it is a
regression test, not evidence of real-world performance.

DOM-1 (extension/test/pages/*.html) is a set of real public pages saved as static snapshots (scripts removed,
stylesheets inlined and pruned, images replaced by same-size blanks, web fonts dropped). Evaluation runs with
the network blocked, at a 1280x800 viewport. Pages with a <name>.gt.json file carry hand-labelled ground truth
for the actionable elements in the first viewport; precision/recall get bootstrap CIs.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

from .metrics import N_BOOT, SEED, _ratio_ci
from .registry import log_run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXT = os.path.join(ROOT, "extension")
TEST = os.path.join(EXT, "test")
PAGES = os.path.join(TEST, "pages")
VIEWPORT = (1280, 800)
MODEL_VERSION = "dom-v1.1"   # keep in step with version in extension/dom-model.js
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/140.0.0.0 Safari/537.36")

CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "google-chrome", "chromium", "chromium-browser"]

# DOM-1 page set: one of each kind of page people use. Snapshots are small, for testing only; content belongs to
# the respective sites (Wikipedia text is CC BY-SA 4.0).
SOURCES = {
    "wikipedia-article": "https://en.wikipedia.org/wiki/Electromyography",
    "wikipedia-main": "https://en.wikipedia.org/wiki/Main_Page",
    "github-repo": "https://github.com/psf/requests",
    "github-login": "https://github.com/login",
    "hackernews": "https://news.ycombinator.com/",
    "hackernews-login": "https://news.ycombinator.com/login",
    "bbc-news": "https://www.bbc.com/news",
    "bing-search": "https://www.bing.com/search?q=electromyography",
    "ddg-search": "https://html.duckduckgo.com/html/?q=electromyography",
    "python-docs": "https://docs.python.org/3/library/json.html",
    "mdn-docs": "https://developer.mozilla.org/en-US/docs/Web/HTML/Element/button",
    "npm-package": "https://www.npmjs.com/package/react",
    "usa-gov": "https://www.usa.gov/",
    "craigslist": "https://sfbay.craigslist.org/",
    "weather-gov": "https://forecast.weather.gov/MapClick.php?lat=37.7749&lon=-122.4194",
    "shop-product": "https://www.allbirds.com/products/mens-tree-runners",
    "books-product": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
    "stackoverflow": "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array",
    "old-reddit": "https://old.reddit.com/r/programming/",
    "arxiv-abs": "https://arxiv.org/abs/2106.01933",
}

# Some pages open a cross-origin consent / marketing iframe over the whole viewport on first visit. Its buttons are
# out of the model's reach (iframes are out of scope), so as captured the model finds 0 targets on these pages.
# For DOM-1 we remove the overlay, as if the person had dismissed it, and record that in SOURCES.json.
DISMISS = {
    "bbc-news": "document.querySelectorAll('[id^=sp_message_container]').forEach(e => e.remove());"
                "document.documentElement.classList.remove('sp-message-open');",
    "shop-product": "document.querySelectorAll('#attentive_overlay').forEach(e => e.remove());"
                    "document.querySelectorAll('body > [aria-hidden=true]').forEach(e => e.removeAttribute('aria-hidden'));",
}


def find_browser():
    for c in CANDIDATES:
        if os.path.exists(c) or shutil.which(c):
            return c
    raise RuntimeError("Chrome or Edge not found")


def file_url(path):
    return "file:///" + os.path.abspath(path).replace("\\", "/")


# ------------------------------------------------------------------ minimal Chrome DevTools Protocol client
class Chrome:
    """Headless Chrome with one page, driven over CDP (websockets). offline=True blocks all network access."""

    def __init__(self, offline=True, viewport=VIEWPORT):
        from websockets.sync.client import connect
        self.prof = tempfile.mkdtemp(prefix="afono-cdp-")
        args = [find_browser(), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                "--allow-file-access-from-files", "--remote-debugging-port=0", "--hide-scrollbars",
                f"--user-data-dir={self.prof}", f"--window-size={viewport[0]},{viewport[1]}", "about:blank"]
        if offline:
            args.insert(1, "--host-resolver-rules=MAP * ~NOTFOUND")
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = os.path.join(self.prof, "DevToolsActivePort")
        for _ in range(200):
            if os.path.exists(port_file) and open(port_file).read().strip():
                break
            time.sleep(0.05)
        port = open(port_file).read().split()[0]
        for _ in range(100):
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
                ws = [p for p in pages if p["type"] == "page"][0]["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.05)
        self.ws = connect(ws, max_size=2 ** 28, open_timeout=20)
        self.id = 0
        self.events = []
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Emulation.setDeviceMetricsOverride", width=viewport[0], height=viewport[1], deviceScaleFactor=1,
                  mobile=False)
        self.send("Network.enable")
        self.send("Network.setUserAgentOverride", userAgent=UA)

    def send(self, method, timeout=60, **params):
        self.id += 1
        mid = self.id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        end = time.time() + timeout
        while time.time() < end:
            msg = json.loads(self.ws.recv(timeout=max(0.1, end - time.time())))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)
        raise TimeoutError(method)

    def navigate(self, url, settle=0.5, timeout=45):
        self.events.clear()
        self.send("Page.navigate", url=url)
        end = time.time() + timeout
        while time.time() < end:
            if any(e.get("method") == "Page.loadEventFired" for e in self.events):
                break
            try:
                self.events.append(json.loads(self.ws.recv(timeout=0.5)))
            except TimeoutError:
                pass
        time.sleep(settle)

    def eval(self, expr, timeout=120):
        r = self.send("Runtime.evaluate", timeout=timeout, expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description")
                               or r["exceptionDetails"].get("text"))
        return r["result"].get("value")

    def inject(self, *files):
        for f in files:
            with open(f, encoding="utf-8") as fh:
                self.eval(fh.read() + "\n;0")

    def screenshot(self, path):
        data = self.send("Page.captureScreenshot", format="png")["data"]
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.kill()
            self.proc.wait(timeout=10)
            shutil.rmtree(self.prof, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def load_model(ch):
    ch.inject(os.path.join(EXT, "dom-model.js"), os.path.join(EXT, "commands.js"), os.path.join(TEST, "score.js"))


# ------------------------------------------------------------------ DOM-0 fixture
def run_fixture(ch=None, log=True, notes=""):
    own = ch is None
    ch = ch or Chrome()
    try:
        ch.navigate(file_url(os.path.join(TEST, "fixture.html")))
        load_model(ch)
        res = ch.eval("AfonoEval.fixture(document)")
    finally:
        if own:
            ch.close()
    if log:
        log_run("evaluate", "dom-0", model=MODEL_VERSION, benchmark="DOM-0 fixture v3", metrics={
            "precision": res["precision"], "recall": res["recall"], "label_accuracy": res["label_accuracy"],
            "name_accuracy": res["name_accuracy"], "commands_ok": res["commands_ok"],
            "commands_total": res["commands_total"], "vocabulary_size": res["vocabulary_size"]},
            details={k: v for k, v in res.items() if k != "targets"}, notes=notes)
    return res


# ------------------------------------------------------------------ DOM-1 real pages
def page_files():
    return sorted(f[:-5] for f in os.listdir(PAGES) if f.endswith(".html")) if os.path.isdir(PAGES) else []


def load_gt(name):
    p = os.path.join(PAGES, name + ".gt.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _cluster_ci(num, den, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over pages (clusters): resample pages, pool their counts."""
    return _ratio_ci(np.asarray(num, float), np.asarray(den, float), n_boot=n_boot, seed=seed)


def run_dom1(ch=None, log=True, notes="", verbose=True):
    names = page_files()
    if not names:
        raise RuntimeError(f"no snapshots in {PAGES}; run `python -m semg.dom_eval snapshot` first")
    own = ch is None
    ch = ch or Chrome()
    per = {}
    try:
        for name in names:
            ch.navigate(file_url(os.path.join(PAGES, name + ".html")), settle=0.3)
            load_model(ch)
            gt = load_gt(name)
            paths = json.dumps([g["path"] for g in gt["targets"]]) if gt else "null"
            per[name] = ch.eval(f"AfonoEval.page(document, {paths})")
            if verbose:
                v, p = per[name]["viewport"], per[name]["page"]
                print(f"  {name:20s} viewport {v['targets']:4d} targets, {v['vocab_labels']:3d} label words | "
                      f"page {p['targets']:5d} targets, {p['vocab_labels']:4d} label words", flush=True)
    finally:
        if own:
            ch.close()
    res = summarize_dom1(per)
    if log:
        log_run("evaluate", "dom-1", model=MODEL_VERSION, benchmark="DOM-1 real pages v1", metrics=res["headline"],
                details={"summary": res, "pages": {k: {kk: vv for kk, vv in v.items() if kk != "targets"}
                                                   for k, v in per.items()}}, notes=notes)
    return res, per


def summarize_dom1(per):
    out = {"n_pages": len(per)}
    for scope in ("viewport", "page"):
        s = [p[scope] for p in per.values()]
        tg = np.array([x["targets"] for x in s], float)
        named = np.array([x["named"] for x in s], float)
        uniq = np.array([x["unique_labels"] for x in s], float)
        vl = np.array([x["vocab_labels"] for x in s], float)
        va = np.array([x["vocab_all"] for x in s], float)
        v1 = np.array([x.get("vocab_labels_one_word", np.nan) for x in s], float)
        out[scope] = {
            "targets_median": float(np.median(tg)), "targets_range": [int(tg.min()), int(tg.max())],
            "named_share": float(named.sum() / max(1, tg.sum())),
            "named_share_ci": _cluster_ci(named, tg),
            "unique_share": float(uniq.sum() / max(1, tg.sum())),
            "label_vocab_median": float(np.median(vl)), "label_vocab_range": [int(vl.min()), int(vl.max())],
            "label_vocab_le30_pages": int((vl <= 30).sum()),
            "full_vocab_median": float(np.median(va)), "full_vocab_range": [int(va.min()), int(va.max())],
            "full_vocab_le30_pages": int((va <= 30).sum()),
            "one_word_label_vocab_median": float(np.median(v1)), "one_word_label_vocab_le30_pages": int((v1 <= 30).sum()),
            "label_vocab_by_page": {k: p[scope]["vocab_labels"] for k, p in per.items()},
        }
    gts = {k: p["gt"] for k, p in per.items() if p.get("gt")}
    if gts:
        tp = np.array([g["tp"] for g in gts.values()], float)
        fp = np.array([g["fp"] for g in gts.values()], float)
        fn = np.array([g["fn"] for g in gts.values()], float)
        # element-level CI: each detection (precision) / each true target (recall) is a unit
        el_p = _ratio_ci(np.r_[np.ones(int(tp.sum())), np.zeros(int(fp.sum()))], np.ones(int(tp.sum() + fp.sum())))
        el_r = _ratio_ci(np.r_[np.ones(int(tp.sum())), np.zeros(int(fn.sum()))], np.ones(int(tp.sum() + fn.sum())))
        out["gt"] = {
            "pages": sorted(gts), "n_gt": int(tp.sum() + fn.sum()), "tp": int(tp.sum()), "fp": int(fp.sum()),
            "fn": int(fn.sum()),
            "precision": float(tp.sum() / max(1, tp.sum() + fp.sum())), "precision_ci_pages": _cluster_ci(tp, tp + fp),
            "precision_ci_elements": el_p,
            "recall": float(tp.sum() / max(1, tp.sum() + fn.sum())), "recall_ci_pages": _cluster_ci(tp, tp + fn),
            "recall_ci_elements": el_r,
            "per_page": {k: {"precision": g["tp"] / max(1, g["tp"] + g["fp"]), "recall": g["tp"] / max(1, g["tp"] + g["fn"]),
                             "n_gt": g["n_gt"]} for k, g in gts.items()},
        }
    v = out["viewport"]
    out["headline"] = {"pages": len(per), "viewport_targets_median": v["targets_median"],
                       "viewport_named_share": v["named_share"], "viewport_unique_share": v["unique_share"],
                       "viewport_label_vocab_median": v["label_vocab_median"],
                       "viewport_label_vocab_le30_share": v["label_vocab_le30_pages"] / max(1, len(per)),
                       "page_label_vocab_median": out["page"]["label_vocab_median"],
                       "page_label_vocab_le30_share": out["page"]["label_vocab_le30_pages"] / max(1, len(per))}
    if gts:
        out["headline"].update({"gt_precision": out["gt"]["precision"], "gt_recall": out["gt"]["recall"]})
    return out


# ------------------------------------------------------------------ snapshots
SERIALIZE_JS = r"""
(async function () {
  const doc = document, win = window;
  // 1. freeze image sizes, then blank them (keeps layout, drops the bytes)
  const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";
  doc.querySelectorAll("img").forEach(img => {
    const r = img.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) { img.style.setProperty("width", r.width + "px", "important"); img.style.setProperty("height", r.height + "px", "important"); }
    img.setAttribute("src", BLANK); img.removeAttribute("srcset"); img.removeAttribute("sizes"); img.removeAttribute("loading");
  });
  doc.querySelectorAll("picture source, video source, audio source, track").forEach(e => e.remove());
  doc.querySelectorAll("video").forEach(v => { v.removeAttribute("src"); v.removeAttribute("poster"); });
  doc.querySelectorAll("iframe").forEach(f => {
    const r = f.getBoundingClientRect();
    f.setAttribute("data-src", (f.getAttribute("src") || "").slice(0, 200)); f.removeAttribute("src"); f.removeAttribute("srcdoc");
    f.style.setProperty("width", r.width + "px"); f.style.setProperty("height", r.height + "px");
  });
  // 2. stylesheets: keep rules whose selector matches something (ignoring states and pseudo-elements)
  function used(sel) {
    const s = sel.replace(/::?(before|after|placeholder|marker|selection|first-line|first-letter|backdrop|file-selector-button|-webkit-[\w-]+|-moz-[\w-]+)(\([^)]*\))?/g, "")
                 .replace(/:(hover|focus|focus-visible|focus-within|active|visited|link|any-link|checked|target|disabled|enabled|placeholder-shown|autofill|user-invalid|invalid|valid)\b/g, "");
    if (!s.trim()) return true;
    try { return !!doc.querySelector(s); } catch (e) { return true; }
  }
  function rules(list) {
    let out = "";
    for (const r of list) {
      if (r.type === 5 || r.type === 7) continue;                          // @font-face, @keyframes
      if (r.type === 1) { if (used(r.selectorText)) out += r.cssText.replace(/url\([^)]*\)/g, "none") + "\n"; continue; }
      if (r.cssRules) {
        const inner = rules(r.cssRules);
        if (inner) { const head = r.cssText.slice(0, r.cssText.indexOf("{")); out += head + "{\n" + inner + "}\n"; }
        continue;
      }
      if (r.type === 3) continue;                                          // @import (inlined via styleSheet)
      out += r.cssText + "\n";
    }
    return out;
  }
  const blocked = [];
  let css = "";
  for (const sh of Array.from(doc.styleSheets).concat(doc.adoptedStyleSheets || [])) {
    if (sh.disabled) continue;
    const media = sh.media && sh.media.mediaText;
    try {
      const t = rules(sh.cssRules);
      css += media && media !== "all" ? "@media " + media + " {\n" + t + "}\n" : t;
    } catch (e) {
      // cross-origin sheet: Python fetches it (no CORS there) and puts it in place of this marker
      if (sh.href) { blocked.push(sh.href); css += "\n/*AFONO_FETCH " + JSON.stringify({ href: sh.href, media: media || "" }) + "*/\n"; }
    }
  }
  // 3. strip scripts, handlers that fetch, and heavy attributes
  doc.querySelectorAll("script, noscript, link[rel=stylesheet], link[rel=preload], link[rel=prefetch], link[rel=modulepreload], link[rel=preconnect], link[rel=dns-prefetch], style, meta[http-equiv], template:not([shadowrootmode]), base").forEach(e => e.remove());
  doc.querySelectorAll("*").forEach(e => {
    for (const a of Array.from(e.attributes)) {
      if (a.name.startsWith("data-") && a.value.length > 120) e.removeAttribute(a.name);
      if (a.name === "style" && /url\(/.test(a.value)) e.setAttribute("style", a.value.replace(/url\([^)]*\)/g, "none"));
    }
  });
  const it = doc.createNodeIterator(doc.documentElement, NodeFilter.SHOW_COMMENT);
  const comments = []; let c; while ((c = it.nextNode())) comments.push(c); comments.forEach(n => n.remove());
  if (window.__afonoDismiss) window.__afonoDismiss();
  const st = doc.createElement("style"); st.textContent = css; doc.head.appendChild(st);
  const roots = []; doc.querySelectorAll("*").forEach(e => { if (e.shadowRoot) roots.push(e.shadowRoot); });
  const html = doc.documentElement.getHTML ? doc.documentElement.getHTML({ serializableShadowRoots: true, shadowRoots: roots }) : doc.documentElement.innerHTML;
  const attrs = Array.from(doc.documentElement.attributes).map(a => a.name + '="' + a.value.replace(/"/g, "&quot;") + '"').join(" ");
  return { html: "<html " + attrs + ">" + html + "</html>", blocked: blocked, title: doc.title, shadow_roots: roots.length, css_kb: Math.round(css.length / 1024) };
})()
"""


def _clean_css(t):
    t = re.sub(r"@font-face\s*\{[^}]*\}", "", t)
    t = re.sub(r"/\*# sourceMappingURL=[^*]*\*/", "", t)
    return re.sub(r"url\([^)]*\)", "none", t)


def _inline_remote_css(html):
    import requests

    def fetch(m):
        spec = json.loads(m.group(1))
        try:
            r = requests.get(spec["href"], headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            t = _clean_css(r.text)
        except Exception as e:
            return f"/* could not fetch {spec['href']}: {e} */"
        return f"@media {spec['media']} {{\n{t}\n}}" if spec["media"] and spec["media"] != "all" else t
    return re.sub(r"/\*AFONO_FETCH (\{.*?\})\*/", fetch, html)


def snapshot(names=None, settle=3.0):
    os.makedirs(PAGES, exist_ok=True)
    names = names or list(SOURCES)
    report = {}
    with Chrome(offline=False) as ch:
        for name in names:
            url = SOURCES[name]
            try:
                ch.navigate(url, settle=settle)
                # click nothing: the page as a first-time visitor sees it, minus a full-screen consent overlay
                if name in DISMISS:                                # run again right before serializing
                    ch.eval(DISMISS[name] + ";window.__afonoDismiss = function () {" + DISMISS[name] + "};0")
                info = ch.eval(SERIALIZE_JS, timeout=180)
                info["html"] = _inline_remote_css(info["html"])
                header = (f"<!doctype html>\n<!-- Afono DOM-1 test snapshot of {url} taken {time.strftime('%Y-%m-%d')} "
                          f"with headless Chrome at {VIEWPORT[0]}x{VIEWPORT[1]}. Scripts removed, CSS inlined and pruned, "
                          f"images blanked. Content belongs to its owners; kept only as a small test fixture. -->\n")
                body = header + info["html"]
                with open(os.path.join(PAGES, name + ".html"), "w", encoding="utf-8") as f:
                    f.write(body)
                report[name] = {"url": url, "title": info["title"], "kb": round(len(body.encode()) / 1024),
                                "css_kb": info["css_kb"], "shadow_roots": info["shadow_roots"],
                                "fetched_css": len(info["blocked"]), "date": time.strftime("%Y-%m-%d"),
                                "overlay_dismissed": name in DISMISS}
                print(f"  {name:20s} {report[name]['kb']:5d} KB  {info['title'][:60]}", flush=True)
            except Exception as e:
                report[name] = {"url": url, "error": str(e)[:200]}
                print(f"  {name:20s} FAILED {e}", flush=True)
    src = os.path.join(PAGES, "SOURCES.json")
    old = json.load(open(src, encoding="utf-8")) if os.path.exists(src) else {}
    old.update(report)
    with open(src, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1)
    return report


def candidates(name, out_dir=None):
    """Screenshot the first viewport of a snapshot with every plausible target boxed (green = detected,
    red = candidate the model skipped), and write the candidate list, for hand annotation."""
    out_dir = out_dir or tempfile.gettempdir()
    with Chrome() as ch:
        ch.navigate(file_url(os.path.join(PAGES, name + ".html")), settle=0.3)
        load_model(ch)
        ch.screenshot(os.path.join(out_dir, name + ".png"))
        cands = ch.eval("AfonoEval.candidates(document)")
        items = [{"rect": c["rect"], "tag": str(i), "color": "#1a9850" if c["detected"] else "#d73027"}
                 for i, c in enumerate(cands)]
        ch.eval(f"AfonoEval.overlay(document, {json.dumps(items)})")
        ch.screenshot(os.path.join(out_dir, name + ".boxes.png"))
    with open(os.path.join(out_dir, name + ".candidates.json"), "w", encoding="utf-8") as f:
        json.dump(cands, f, indent=1)
    return cands


# ------------------------------------------------------------------ CLI
def _print_fixture(r):
    print("DOM-0 fixture:", {k: r[k] for k in ("targets_found", "ground_truth", "true_positives", "precision", "recall",
                                               "label_accuracy", "name_checks", "name_accuracy", "labels_unique",
                                               "commands_ok", "commands_total", "vocabulary_size")})
    for m in r["misses"]:
        print("   ", m)


def _pct(x):
    return f"{100 * x:.1f}%"


def _print_dom1(res):
    print(f"DOM-1: {res['n_pages']} real pages (1280x800, offline snapshots)")
    for scope in ("viewport", "page"):
        s = res[scope]
        print(f"  [{scope}] targets median {s['targets_median']:.0f} (range {s['targets_range']}), "
              f"named {_pct(s['named_share'])} [{_pct(s['named_share_ci'][0])}, {_pct(s['named_share_ci'][1])}], "
              f"unique {_pct(s['unique_share'])}, label vocab median {s['label_vocab_median']:.0f} "
              f"(range {s['label_vocab_range']}), <=30 words on {s['label_vocab_le30_pages']}/{res['n_pages']} pages; "
              f"with commands+numbers median {s['full_vocab_median']:.0f}; one-word labels median "
              f"{s['one_word_label_vocab_median']:.0f}, <=30 on {s['one_word_label_vocab_le30_pages']}/{res['n_pages']}")
    if "gt" in res:
        g = res["gt"]
        print(f"  [ground truth, {len(g['pages'])} pages, {g['n_gt']} targets] "
              f"precision {_pct(g['precision'])} pages-CI [{_pct(g['precision_ci_pages'][0])}, {_pct(g['precision_ci_pages'][1])}] "
              f"elements-CI [{_pct(g['precision_ci_elements'][0])}, {_pct(g['precision_ci_elements'][1])}]; "
              f"recall {_pct(g['recall'])} pages-CI [{_pct(g['recall_ci_pages'][0])}, {_pct(g['recall_ci_pages'][1])}] "
              f"elements-CI [{_pct(g['recall_ci_elements'][0])}, {_pct(g['recall_ci_elements'][1])}]")


def run(notes=""):
    """Everything with one command: DOM-0 fixture and DOM-1 real pages."""
    with Chrome() as ch:
        fx = run_fixture(ch, notes=notes)
        d1 = run_dom1(ch, notes=notes) if page_files() else (None, None)
    return fx, d1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    notes = os.environ.get("AFONO_NOTES", "DOM Model v1 (accname + short labels)")
    if cmd == "snapshot":
        snapshot(sys.argv[2:] or None)
    elif cmd == "candidates":
        out = sys.argv[3] if len(sys.argv) > 3 else None
        for c in candidates(sys.argv[2], out):
            pass
        print("wrote screenshots and candidates to", out or tempfile.gettempdir())
    elif cmd == "fixture":
        _print_fixture(run_fixture(notes=notes, log="--no-log" not in sys.argv))
    elif cmd == "dom1":
        res, per = run_dom1(notes=notes, log="--no-log" not in sys.argv)
        _print_dom1(res)
        for k, p in per.items():
            if p.get("gt"):
                print(f"   {k}: FP {p['gt']['fp_items'][:8]}  FN {p['gt']['fn_items'][:8]}")
    else:
        fx, (res, per) = run(notes)
        _print_fixture(fx)
        if res:
            _print_dom1(res)
