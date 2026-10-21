"""Build the website's progress data from the run registry.

    python -m semg.site_data

Writes site/data/progress.json and site/data/progress.js (the .js copy lets the site
work when opened straight from disk). Numbers only ever come from runs/registry.jsonl
and site/demo/samples.json, so the site shows what was measured and nothing else.
"""
import datetime as _dt
import json
import os

from .registry import read_runs

SITE = os.environ.get("AFONO_SITE", "site")


def _pct(x):
    return None if x is None else round(100 * x, 1)


def _summary(r):
    k = r["kind"]
    if k == "train":
        n = r.get('train_utterances')
        return (f"Trained {r.get('params_m', '?')}M-parameter model on {f'{n:,}' if isinstance(n, int) else '?'} recordings "
                f"for {r.get('hours', '?')} h. Dev character error {_pct(r.get('dev_cer'))}%.")
    if k == "evaluate" and r.get("model") == "voice":
        m = r.get("metrics", {})
        return (f"Whisper mode {r.get('benchmark')}: {_pct(m.get('accuracy'))}% of prompts right"
                + (f" over {m['trials']} trials" if "trials" in m else "") + ".")
    if k == "evaluate" and r.get("model") == "intent":
        m = r.get("metrics", {})
        if "sus" in m:
            return f"Usability survey ({r.get('participant')}): SUS {m['sus']:.1f} of 100."
        if "tasks" in m:
            return (f"{r.get('benchmark')} by {r.get('participant')}: {_pct(m.get('success_rate'))}% of {m['tasks']} tasks done, "
                    f"median {m.get('median_s')} s per task, {m.get('errors')} errors, {m.get('undos')} undos.")
        return f"{r.get('benchmark')}: " + ", ".join(f"{a} {b}" for a, b in m.items()) + "."
    if k == "evaluate" and r.get("model") not in (None, "dom", "track"):
        return f"{r.get('benchmark') or r['run_id']}."
    if k == "evaluate" and r.get("model") is None:
        m = r.get("metrics", {})
        return (f"Benchmark ({r.get('lm', 'unigram')} decoder): dates & times {_pct(m.get('dates_word_acc'))}% word accuracy, "
                f"500-word sentences {_pct(m.get('closed500_word_acc'))}%, open vocabulary WER {_pct(m.get('open_wer'))}%.")
    if k == "evaluate" and r.get("model") == "dom":
        m = r.get("metrics", {})
        return (f"DOM Model on {r.get('benchmark')}: precision {_pct(m.get('precision'))}%, recall {_pct(m.get('recall'))}%, "
                f"labels right {_pct(m.get('label_accuracy'))}%.")
    if k == "evaluate" and r.get("model") == "track":
        m = r.get("metrics", {})
        return f"Track Model {r.get('benchmark')}: " + (f"{m.get('accuracy_deg', 0):.2f}° accuracy." if "accuracy_deg" in m else f"{_pct(m.get('hit_rate'))}% hits.")
    if k == "compare" and r.get("task"):
        return (f"Decoder comparison, {r['task']}: WER {_pct(r.get('wer_a'))}% to {_pct(r.get('wer_b'))}% "
                f"(95% CI {_pct(r['delta_ci95'][0])} to {_pct(r['delta_ci95'][1])} pts), p = {r.get('p_b_not_better'):.3f}.")
    if k == "compare":
        return (f"Paired test on {r.get('n_utterances')} held-out recordings: WER {_pct(r.get('wer_a'))}% to "
                f"{_pct(r.get('wer_b'))}%, change {_pct(r.get('delta_wer'))} pts (95% CI {_pct(r['delta_ci95'][0])} "
                f"to {_pct(r['delta_ci95'][1])}), p = {r.get('p_b_not_better'):.2f}.")
    if k == "calibrate":
        b, a = r.get("before") or {}, r.get("after") or {}
        return (f"Calibrated on {r.get('train_recordings')} labelled recordings: held-out word accuracy "
                f"{_pct(b.get('lexicon_word_acc'))}% to {_pct(a.get('lexicon_word_acc'))}%.")
    return k


def _report(r):
    p = (r.get("report") or "").replace("\\", "/")
    if p and os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _best(evals):
    """The run the site reports: best page30-hard word accuracy under the newest protocol, not simply the latest
    run (a failed run must not replace a good one). Protocol v2 runs, which lack page tasks, rank by closed500."""
    for proto, task in (("v3", "page30-hard"), ("v2", "closed500")):
        cand = []
        for r in evals:
            if r.get("protocol") != proto:
                continue
            t = _report(r).get("tasks", {}).get(task)
            if t:
                cand.append((t["lexicon"]["word_accuracy"], r["time"], r))
        if cand:
            return max(cand, key=lambda c: (c[0], c[1]))[2]
    return None


def build(site=SITE, registry=None):
    runs = read_runs(registry)
    evals = [r for r in runs if r["kind"] == "evaluate" and r.get("model") is None]
    best = _best(evals)
    latest = best or (evals[-1] if evals else None)
    demo = {}
    sj = os.path.join(site, "demo", "samples.json")
    if os.path.exists(sj):
        with open(sj) as f:
            s = json.load(f)
        mp = os.path.join(site, "demo", "model.onnx")
        demo = {"model": s.get("model"), "int8_agreement": s.get("int8_argmax_agreement"),
                "model_mb": round(os.path.getsize(mp) / 1e6, 1) if os.path.exists(mp) else None,
                "samples": len(s.get("samples", []))}
    roadmap = {}
    rp = os.path.join(site, "data", "roadmap.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            roadmap = json.load(f)
    m = (latest or {}).get("metrics", {})
    research = {}
    if best:
        r = best
        rep = _report(r)
        order = [("dates", "Dates and times"), ("closed100", "Book sentences"), ("closed250", "Book sentences"),
                 ("closed500", "Book sentences"), ("open", "Any sentence"), ("page30-hard", "Page words, look-alike distractors")]
        rows = []
        for key, label in order:
            t = rep.get("tasks", {}).get(key)
            if not t:
                continue
            L, G = t["lexicon"], t["greedy"]
            rows.append({"task": key, "label": label, "vocab": L.get("vocab_size"), "n": L["n_utterances"],
                         "words": L["n_words"], "acc": _pct(L["word_accuracy"]),
                         "acc_ci": [_pct(x) for x in L["word_accuracy_ci95"]], "wer": _pct(L["wer"]),
                         "greedy_acc": _pct(G["word_accuracy"]), "bits": round(L.get("bits_per_word", 0), 2),
                         "itr": round(L.get("itr_bits_per_min", 0), 1), "cwpm": round(L.get("correct_wpm", 0), 1),
                         "rtf": round(L.get("real_time_factor", 0), 3)})
        research = {"run": r["run_id"], "date": r["time"][:10], "protocol": r.get("protocol"), "lm": r.get("lm", "unigram"), "tasks": rows,
                    "dataset": rep.get("dataset"), "decoder": rep.get("decoder"), "compute": rep.get("compute"),
                    "params_m": rep.get("params_m"), "epoch": rep.get("epoch")}
    calib = [r for r in runs if r["kind"] == "compare" and r.get("profile")]
    if calib:
        c = calib[-1]
        research["paired"] = {k: c.get(k) for k in ("wer_a", "wer_b", "delta_wer", "delta_ci95", "p_b_not_better",
                                                    "n_utterances", "profile", "a", "b")}
    # the site's decoder ablation table is the unigram-vs-bigram study; other paired tests stay in the run log
    dec = [r for r in runs if r["kind"] == "compare" and r.get("task") and "unigram vs bigram" in r.get("notes", "").lower()]
    if dec:
        last = {}
        for r in dec:
            last[r["task"]] = r
        research["decoder_compare"] = [{"task": t, "wer_a": _pct(r["wer_a"]), "wer_b": _pct(r["wer_b"]),
                                        "delta": _pct(r["delta_wer"]), "ci": [_pct(x) for x in r["delta_ci95"]],
                                        "p": round(r["p_b_not_better"], 3), "n": r["n_utterances"], "notes": r.get("notes", "")}
                                       for t, r in sorted(last.items())]
    dom = [r for r in runs if r.get("model") == "dom"]
    if dom:
        research["dom"] = {"date": dom[-1]["time"][:10], "benchmark": dom[-1].get("benchmark"), **dom[-1].get("metrics", {}),
                           "targets": dom[-1].get("details", {}).get("ground_truth")}
    track = [r for r in runs if r.get("model") == "track"]
    if track:
        research["track"] = [{"date": r["time"][:10], "benchmark": r.get("benchmark"), **r.get("metrics", {})} for r in track[-5:]]
    if research.get("tasks"):
        t = {row["task"]: row for row in research["tasks"]}
        m = dict(m)
        for key, name in (("dates", "dates_word_acc"), ("closed100", "closed100_word_acc"), ("closed500", "closed500_word_acc")):
            if key in t:
                m[name] = t[key]["acc"] / 100
        if "open" in t:
            m["open_wer"] = t["open"]["wer"] / 100
    data = {
        "updated": _dt.datetime.now().astimezone().isoformat(timespec="minutes"),
        "headline": {
            "dates_word_acc": _pct(m.get("dates_word_acc")),
            "closed500_word_acc": _pct(m.get("closed500_word_acc")),
            "closed100_word_acc": _pct(m.get("closed100_word_acc")),
            "open_wer": _pct(m.get("open_wer")),
            "from_run": latest["run_id"] if latest else None,
            "measured": latest["time"][:10] if latest else None,
        },
        "demo": demo,
        "research": research,
        "runs": [{"date": r["time"][:10], "kind": r["kind"], "run_id": r["run_id"], "summary": _summary(r),
                  "notes": r.get("notes", "")} for r in reversed(runs)],
        "counts": {k: sum(1 for r in runs if r["kind"] == k) for k in ("train", "calibrate", "evaluate")},
        **roadmap,
    }
    os.makedirs(os.path.join(site, "data"), exist_ok=True)
    with open(os.path.join(site, "data", "progress.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(site, "data", "progress.js"), "w", encoding="utf-8") as f:
        f.write("window.AFONO_PROGRESS = " + json.dumps(data) + ";\n")
    return data


def main():
    d = build()
    print(f"site data updated: {len(d['runs'])} runs, headline from run '{d['headline']['from_run']}'")


if __name__ == "__main__":
    main()
