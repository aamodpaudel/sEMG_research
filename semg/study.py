"""User study: the hands-free task list, its mouse baseline and a short survey, per participant.

The YC demo protocol (PLAN.md, "The YC demo"): someone sits at the laptop, hands in lap, and works through a fixed
task list by voice (and later gaze or silent speech); the same tasks with a mouse are the baseline. We time every
task and count errors (a wrong action) and undos. Then the System Usability Scale and a few open questions.

    python -m semg.study new --name "Aamod"                  # consent + a pseudonym (p01, p02, ...)
    python -m semg.study tasks                               # the task list and what to open first
    python -m semg.study run --pid p01 --mode mouse          # timed task list, observer presses Enter
    python -m semg.study run --pid p01 --mode whisper --device "AirPods Pro"
    python -m semg.study survey --pid p01                    # SUS + open questions
    python -m semg.study summary [--log]                     # per person and mode, hands-free vs mouse

Privacy: names, health notes, quotes and per-task rows stay in data/study/ (git-ignored, this computer only).
runs/registry.jsonl, which is public, only ever gets the pseudonym and the numbers.
"""
import argparse
import datetime as _dt
import glob
import json
import os
import statistics

from .registry import log_run

ROOT = os.environ.get("AFONO_STUDY_ROOT", os.path.join("data", "study"))
TASK_SET = "demo_v1"
START_PAGES = ["https://news.ycombinator.com"]
# Every task stays on Hacker News pages, so the extension runs on each page and every step can be done by voice
# (Nullpoint cannot switch browser tabs, and a story link can open a PDF where the extension does not run).
TASKS = [
    "Open the comments of the third story.",
    "Scroll down twice.",
    "Go back to the front page.",
    "Open the 'new' page from the menu at the top.",
    "Zoom in, then zoom out.",
    "Go to the bottom of the page and open 'Guidelines'.",
    "Go back.",
]
MODES = ["mouse", "whisper", "whisper-gaze", "gaze", "silent", "silent-gaze"]
GROUPS = ["none", "hand-pain-or-rsi", "autoimmune", "limited-hand-use", "voice-loss", "prefer-not-to-say"]

# System Usability Scale (Brooke 1996). Odd items are positive, even items negative.
SUS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]
OPEN = [
    ("best", "What was the best moment?"),
    ("worst", "What was the most frustrating moment?"),
    ("daily", "What would make you use this every day?"),
    ("site", "The website (afono): first impression in a few words, and anything confusing?"),
]


def _now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _people_path(root):
    return os.path.join(root, "participants.json")


def load_people(root=ROOT) -> dict:
    p = _people_path(root)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save_people(people, root=ROOT):
    os.makedirs(root, exist_ok=True)
    with open(_people_path(root), "w", encoding="utf-8") as f:
        json.dump(people, f, indent=2)


def next_pid(people) -> str:
    n = max([int(k[1:]) for k in people if k[1:].isdigit()] + [0]) + 1
    return f"p{n:02d}"


def sus_score(answers) -> float:
    """answers: 10 ratings, 1 (strongly disagree) to 5 (strongly agree). Returns 0-100."""
    if len(answers) != 10 or any(a not in (1, 2, 3, 4, 5) for a in answers):
        raise ValueError("SUS needs 10 answers from 1 to 5")
    return 2.5 * sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))


def task_metrics(rows) -> dict:
    done = [r for r in rows if r["result"] == "done"]
    times = [r["seconds"] for r in done]
    return {
        "tasks": len(rows),
        "success_rate": round(len(done) / len(rows), 3) if rows else None,
        "median_s": round(statistics.median(times), 1) if times else None,
        "total_s": round(sum(r["seconds"] for r in rows), 1),
        "errors": sum(r["errors"] for r in rows),
        "undos": sum(r["undos"] for r in rows),
    }


def sessions(root=ROOT) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, "p*", "*.json"))):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        s["_path"] = path
        out.append(s)
    return out


def summarize(root=ROOT) -> dict:
    """Latest task run per (participant, mode), latest SUS per participant, and how each hands-free mode compares
    with the same person's mouse run (ratio of total time; done tasks only count when both finished them)."""
    runs, sus = {}, {}
    for s in sessions(root):
        if s.get("type") == "tasks":
            runs[(s["pid"], s["mode"])] = s
        elif s.get("type") == "survey":
            sus[s["pid"]] = s["sus"]
    rows, ratios = [], {}
    for (pid, mode), s in sorted(runs.items()):
        m = task_metrics(s["tasks"])
        row = {"pid": pid, "mode": mode, **m, "sus": sus.get(pid)}
        base = runs.get((pid, "mouse"))
        if mode != "mouse" and base:
            pairs = [(a["seconds"], b["seconds"]) for a, b in zip(s["tasks"], base["tasks"])
                     if a["result"] == "done" and b["result"] == "done"]
            if pairs and sum(b for _, b in pairs) > 0:
                row["vs_mouse"] = round(sum(a for a, _ in pairs) / sum(b for _, b in pairs), 2)
                ratios.setdefault(mode, []).append(row["vs_mouse"])
        rows.append(row)
    return {"rows": rows, "median_vs_mouse": {k: statistics.median(v) for k, v in ratios.items()},
            "people": len({r["pid"] for r in rows} | set(sus))}


# ------------------------------------------------------------------ interactive parts (observer at the keyboard)

def _ask(prompt, default=None, cast=str, choices=None):
    while True:
        raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if not raw and default is not None:
            raw = str(default)
        try:
            v = cast(raw)
        except ValueError:
            print("  Please try again.")
            continue
        if choices and v not in choices:
            print("  Choose one of: " + ", ".join(map(str, choices)))
            continue
        return v


def cmd_new(args):
    people = load_people(args.root)
    pid = next_pid(people)
    print(f"\nNew participant {pid}. Read this to them:\n"
          "  We are testing Afono Nullpoint, software that lets you use a computer without your hands.\n"
          "  We time a few tasks, with a mouse and hands-free, and ask a few questions. It takes about 25 minutes.\n"
          "  Your name stays on this computer only; results are published under the code " + pid + ".\n"
          "  If you choose, short voice clips of the test prompts are kept on this computer to improve the model.\n"
          "  You can stop at any time, and ask for your data to be deleted.\n")
    if _ask("Do they agree to take part? (y/n)", choices=["y", "n"]) != "y":
        print("Not added.")
        return
    clips = _ask("May we keep their voice clips on this computer? (y/n)", choices=["y", "n"]) == "y"
    print("Optional, self-described, kept on this computer only: " + ", ".join(GROUPS))
    group = _ask("Group", default="prefer-not-to-say", choices=GROUPS)
    people[pid] = {"name": args.name, "added": _now(), "consent": True, "keep_voice_clips": clips, "group": group}
    _save_people(people, args.root)
    first = "mouse" if int(pid[1:]) % 2 else "whisper"
    print(f"\nAdded {pid}. Use the profile name {pid} in Nullpoint too.")
    print(f"Order for {pid}: {first} first, then {'whisper' if first == 'mouse' else 'mouse'} (alternates per person).")


def cmd_tasks(args):
    print(f"Task list {TASK_SET}. Before each run, open this page in one tab and nothing else:")
    for u in START_PAGES:
        print("  " + u)
    print("Scrolled to the top, zoom at 100% (Ctrl+0), number hints off.\n")
    for i, t in enumerate(TASKS, 1):
        print(f"  {i}. {t}")


def cmd_run(args):
    people = load_people(args.root)
    if args.pid not in people:
        raise SystemExit(f"Unknown participant {args.pid}. Add them with: python -m semg.study new --name ...")
    cmd_tasks(args)
    print(f"\n{args.pid}, mode {args.mode}. Read each task aloud. Press Enter when they start, Enter when it is done."
          "\nType f then Enter if they give up, s to skip. Wrong actions count as errors, 'undo' or 'back' to fix one"
          " counts as an undo.\n")
    rows = []
    for i, task in enumerate(TASKS, 1):
        input(f"Task {i}: {task}\n  Enter to start...")
        t0 = _dt.datetime.now()
        end = input("  ...Enter when done (f = gave up, s = skip): ").strip().lower()
        secs = (_dt.datetime.now() - t0).total_seconds()
        result = {"f": "failed", "s": "skipped"}.get(end, "done")
        errors = _ask("  Wrong actions", default=0, cast=int)
        undos = _ask("  Undos", default=0, cast=int)
        rows.append({"task": i, "text": task, "seconds": round(secs, 1), "result": result,
                     "errors": errors, "undos": undos})
        print(f"  {result}, {secs:.1f} s")
    notes = input("Observer notes (stay on this computer): ").strip()
    rec = {"type": "tasks", "pid": args.pid, "mode": args.mode, "task_set": TASK_SET, "device": args.device,
           "style": args.style,
           "time": _now(), "tasks": rows, "notes": notes}
    d = os.path.join(args.root, args.pid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{args.mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    m = task_metrics(rows)
    log_run("evaluate", "demo-1", model="intent", benchmark=f"Task list {TASK_SET} ({args.mode})", metrics=m,
            participant=args.pid, mode=args.mode, task_set=TASK_SET, device=args.device[:60], style=args.style)
    print(f"\nSaved {path} and logged to the registry: {m}")


def cmd_survey(args):
    if args.pid not in load_people(args.root):
        raise SystemExit(f"Unknown participant {args.pid}.")
    print("Ask them to answer from 1 (strongly disagree) to 5 (strongly agree), about the hands-free mode.\n")
    answers = [_ask(f"{i}. {q}", cast=int, choices=[1, 2, 3, 4, 5]) for i, q in enumerate(SUS, 1)]
    would = _ask("If using a mouse or keyboard hurt, how likely are you to use this? (1-5)", cast=int,
                 choices=[1, 2, 3, 4, 5])
    open_answers = {k: input(q + " ").strip() for k, q in OPEN}
    score = sus_score(answers)
    rec = {"type": "survey", "pid": args.pid, "time": _now(), "sus_answers": answers, "sus": score,
           "would_use_if_hurt": would, **open_answers}
    d = os.path.join(args.root, args.pid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-survey.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    log_run("evaluate", "study-sus", model="intent", benchmark="System Usability Scale",
            metrics={"sus": score, "would_use_if_hurt": would}, participant=args.pid)
    print(f"\nSUS {score:.1f} / 100 (68 is average). Saved {path}; answers to open questions stay on this computer.")


def cmd_summary(args):
    s = summarize(args.root)
    if not s["rows"]:
        print("No sessions yet.")
        return
    head = f"{'who':5} {'mode':13} {'done':>5} {'median s':>9} {'total s':>8} {'errors':>7} {'undos':>6} {'vs mouse':>9} {'SUS':>5}"
    print(head + "\n" + "-" * len(head))
    for r in s["rows"]:
        print(f"{r['pid']:5} {r['mode']:13} {r['success_rate'] * 100:4.0f}% {r['median_s'] or 0:9.1f} {r['total_s']:8.1f} "
              f"{r['errors']:7d} {r['undos']:6d} {str(r.get('vs_mouse', '')):>9} {str(r['sus'] or ''):>5}")
    for mode, v in s["median_vs_mouse"].items():
        print(f"\n{mode}: median {v:.2f}x the same person's mouse time (1.00 = as fast as a mouse).")
    if args.log:
        log_run("evaluate", "demo-summary", model="intent", benchmark=f"Task list {TASK_SET} summary",
                metrics={"people": s["people"], **{f"{k}_vs_mouse": v for k, v in s["median_vs_mouse"].items()}})
        print("Logged the summary to the registry.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="where private study data lives (default data/study)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new"); p.add_argument("--name", required=True)
    sub.add_parser("tasks")
    p = sub.add_parser("run"); p.add_argument("--pid", required=True); p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--device", default="", help="microphone or earbuds used, e.g. 'AirPods Pro' or 'laptop mic'")
    p.add_argument("--style", default="", choices=["", "whisper", "quiet", "normal"],
                   help="how they spoke in a voice run: whisper (no voice), quiet voice or normal voice")
    p = sub.add_parser("survey"); p.add_argument("--pid", required=True)
    p = sub.add_parser("summary"); p.add_argument("--log", action="store_true")
    args = ap.parse_args()
    {"new": cmd_new, "tasks": cmd_tasks, "run": cmd_run, "survey": cmd_survey, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
