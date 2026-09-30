"""WebUI data loader stub for the Afono DOM Model's future learned detector (stage 2: pages built from unlabeled
clickable <div>s, where accessibility semantics are missing).

WebUI (Wu et al., CHI 2023, "WebUI: A Dataset for Enhancing Visual UI Understanding with Web Semantics")
  code + download scripts  https://github.com/js0nwu/webui   (project page https://uimodeling.github.io/)
  HuggingFace              https://huggingface.co/datasets?search=biglab/webui
  size (HF, checked 2026-09-29): ~400K pages crawled; splits webui-7k 8.4 GB, webui-70k 72.6 GB, webui-350k 465 GB,
    webui-val 8.8 GB, webui-test 17.2 GB, webui-all 550 GB (full: screenshots per device, HTML, AX tree, boxes, styles).
    Repacked "-elements" parquet (screenshot + boxes + AX-role labels only): 7k 1.3 GB, val 1.4 GB, test 2.7 GB,
    70k 11.4 GB, 350k 68 GB. Splits are by domain (70/10/20), so test sites are unseen.
  license: "other" - CMU research terms in COPYRIGHT.txt (no warranty; researcher indemnifies CMU; screenshots may
    contain copyrighted work; a commercial employer is bound too). No explicit commercial-use grant: check before
    shipping weights trained on it in a product. The HF release was filtered with a bad-words list, so it has
    fewer pages than the paper.

Per page and device (e.g. key "default_1280-720") the full dataset has: -axtree.json.gz (Chrome AX tree, nodes carry
backendDOMNodeId, role, name), -bb.json.gz (backendDOMNodeId -> {x,y,width,height}, page coordinates), -box.json.gz
(content/padding/border/margin quads), -class.json.gz (CSS classes), -style.json.gz (computed styles incl. cursor),
-viewport.json.gz (in first viewport?), -html.html, -links.json, -url.txt, -screenshot.webp, -screenshot-full.webp.

Important caveat for our use: WebUI's element labels come from the AX tree, i.e. from the same web semantics our rules
already read. A div with a JS click handler is "generic" there too. What the data buys us is a *visual* (and style)
model trained on well-labelled sites that can transfer to badly labelled ones - the paper's own premise. We must
therefore build our own small test set of div-soup pages (hand-labelled, like DOM-1) to measure the gain; WebUI's test
split alone cannot show it.

PLAN (stage 2, after DOM-1 rules are solid)
  1. Data: stream webui-7k-elements (1.3 GB) for a first model; webui-val-elements for model selection. Never
     download 70k/350k until 7k shows a gain. Positive class = AX roles in CLICKABLE_ROLES; everything else negative.
  2. Features, two model options:
     a. DOM-feature classifier (cheap, runs in the content script): per element box geometry, cursor, display,
        background/border contrast, class-token hashes ("btn", "button", "link", "card", "icon"), text length,
        has-img/svg child, depth, sibling repetition. Gradient-boosted trees -> export thresholds/JS. Needs the full
        (non -elements) data for styles/classes: use the 7k split only.
     b. Screenshot detector (heavier, runs in Nullpoint desktop app): small object detector (e.g. SSD/YOLO-n style,
        or the WebUI "screenrecognition" checkpoints as a start) on 1280x720 screenshots -> boxes -> matched to DOM
        elements by IoU in the extension.
  3. Integration: rules first; the model only proposes *extra* targets among non-semantic elements
     (source "learned"), with a precision threshold chosen on val (target >= 90% precision), so it cannot hurt
     semantic pages. Labels for learned targets come from visible text, else "button N".
  4. Evaluation: (i) WebUI test split, element-level P/R of clickable detection vs. rules-only (rules will score high
     there by construction); (ii) DOM-2: 10-20 real div-soup pages, hand-labelled, P/R with bootstrap CIs vs. v1.1.
     Report both; only (ii) answers the product question.

This file only defines the loading side; nothing here downloads more than the one public GitHub sample page.
"""
import gzip
import json
import os
import urllib.request

CLICKABLE_ROLES = {"link", "button", "checkbox", "radio", "switch", "tab", "menuitem", "menuitemcheckbox",
                   "menuitemradio", "option", "treeitem", "combobox", "searchbox", "textbox", "slider", "spinbutton",
                   "PopUpButton", "ToggleButton", "ListBoxOption", "MenuListPopup"}
SAMPLE_URL = "https://raw.githubusercontent.com/js0nwu/webui/main/sample/1656554031731/"
SAMPLE_PARTS = ["axtree.json.gz", "bb.json.gz", "class.json.gz", "style.json.gz", "viewport.json.gz", "url.txt"]


def _load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return f.read() if path.endswith(".txt") else json.load(f)


def fetch_github_sample(out_dir, key="default_1280-720"):
    """Download the single sample page from the WebUI GitHub repo (~40 KB without screenshots)."""
    os.makedirs(out_dir, exist_ok=True)
    for part in SAMPLE_PARTS:
        dst = os.path.join(out_dir, f"{key}-{part}")
        if not os.path.exists(dst):
            urllib.request.urlretrieve(SAMPLE_URL + f"{key}-{part}", dst)
    return out_dir


def load_page(page_dir, key="default_1280-720"):
    """One WebUI page/device as a list of elements:
    {id, box: [x, y, w, h] (page px), role, name, clickable, in_viewport, cursor, classes}.
    Only elements with a bounding box are returned; role comes from the AX tree via backendDOMNodeId."""
    p = lambda part: os.path.join(page_dir, f"{key}-{part}")
    bb = _load(p("bb.json.gz"))
    ax = _load(p("axtree.json.gz"))["nodes"]
    classes = _load(p("class.json.gz")) if os.path.exists(p("class.json.gz")) else {}
    style = _load(p("style.json.gz")) if os.path.exists(p("style.json.gz")) else {}
    vp = _load(p("viewport.json.gz")) if os.path.exists(p("viewport.json.gz")) else {}
    role, name = {}, {}
    for n in ax:
        bid = n.get("backendDOMNodeId")
        if bid is None or n.get("ignored"):
            continue
        role[str(bid)] = n.get("role", {}).get("value", "")
        name[str(bid)] = (n.get("name") or {}).get("value", "")
    out = []
    for bid, b in bb.items():
        if not b or b.get("width", 0) < 1 or b.get("height", 0) < 1:
            continue
        r = role.get(bid, "")
        out.append({"id": bid, "box": [b["x"], b["y"], b["width"], b["height"]], "role": r, "name": name.get(bid, ""),
                    "clickable": r in CLICKABLE_ROLES, "in_viewport": bool(vp.get(bid, False)),
                    "cursor": (style.get(bid) or {}).get("cursor", ""),
                    "classes": list((classes.get(bid) or {}).values())})
    return out


def detection_targets(elements, viewport_only=True):
    """Boxes and 0/1 labels for a clickable-element detector."""
    els = [e for e in elements if e["in_viewport"] or not viewport_only]
    return [e["box"] for e in els], [int(e["clickable"]) for e in els]


def iter_hf_elements(split="webui-7k-elements", streaming=True):
    """Stream the repacked HF element split (screenshot + boxes + AX-role labels) without downloading it whole.
    Needs `pip install datasets`. Box coordinates need a per-device multiplier (see the WebUI repo,
    models/screenrecognition/ui_datasets.py) before use as detector targets."""
    from datasets import load_dataset  # optional dependency
    ds = load_dataset(f"biglab/{split}", split="train", streaming=streaming)
    for row in ds:
        labels = row["labels"]
        yield {"key": row["key_name"], "image": row["image"], "boxes": row["borderBoxes"],
               "clickable": [int(any(r in CLICKABLE_ROLES for r in lab)) for lab in labels], "labels": labels}


if __name__ == "__main__":
    import sys
    import tempfile
    d = fetch_github_sample(sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "webui-sample"))
    els = load_page(d)
    vis = [e for e in els if e["in_viewport"]]
    print(f"{_load(os.path.join(d, 'default_1280-720-url.txt')).strip()}: {len(els)} boxed elements, {len(vis)} in "
          f"viewport, {sum(e['clickable'] for e in vis)} clickable in viewport, "
          f"{sum(e['cursor'] == 'pointer' for e in vis)} with pointer cursor")
