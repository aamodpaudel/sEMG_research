"""Tier 1 command words: record, train and evaluate a 20-30 word silent-command classifier.

Protocol (vocab/commands.json, "recording_protocol"): each item x 10 repetitions x 3 sessions,
randomised order, electrodes re-seated between sessions; sessions 1-2 train, session 3 test.
We also report leave-one-session-out (LOSO) accuracy. Never mix a session across train and test.

    python -m semg.commands plan                                   # the starter set and one session's prompt order
    python -m semg.commands record --user me --session 1 --port COM5   # prompted recording from the ESP32 rig
    python -m semg.commands tag --user me --session 1 --since 2026-10-03T10:00   # tag Nullpoint recordings
    python -m semg.commands train --user me                        # sessions 1-2 -> model, scored on session 3
    python -m semg.commands eval --user me                         # official split + LOSO, logged to the registry
    python -m semg.commands import-gaddy --user gaddy-cmds         # readiness test: words cut from Gaddy recordings
    python -m semg.commands synth --user synth-cmds                # readiness test: synthetic 4-channel recordings

Model: the pretrained silent-speech encoder (semg.model.EMGCTC) with its input layer re-initialised
for the rig's channel count (adapt_channels), an attention-pooling word classifier on top, and
the character CTC head kept as an auxiliary loss on the item's spelling. A feature + shrinkage-LDA
classifier is reported alongside as a sanity baseline.

Recordings live in the user's profile (semg.user.Profile): recordings/<rid>_emg.npy + labels.jsonl.
Session and repetition metadata go to data/user/<id>/commands.jsonl.
"""
import argparse
import datetime as _dt
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EMGCTC, adapt_channels, load_model
from .registry import log_run
from .signal import design_filters
from .text import encode, normalize_text
from . import user as _user


def Profile(name):  # noqa: N802  (profile root can be redirected with AFONO_USER_ROOT, e.g. in tests)
    return _user.Profile(name, root=os.environ.get("AFONO_USER_ROOT", _user.ROOT))

COMMANDS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vocab", "commands.json")
FS = 1000
WINDOW_S = 2.5          # prompt window: 1 s countdown, then 2.5 s capture (mouth the item within ~2 s)


# ---------------------------------------------------------------- vocabulary and metadata
def load_commands(path=COMMANDS_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


ITEM_SETS = ("tier1_starter", "intent_v1")


def starter_items(path=COMMANDS_FILE, set_name="tier1_starter"):
    return list(load_commands(path)[set_name]["items"])


def meta_path(profile):
    return os.path.join(profile.dir, "commands.jsonl")


def add_meta(profile, rid, item, session, rep=None):
    rec = {"rid": rid, "item": normalize_text(item), "session": int(session), "rep": rep,
           "time": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    with open(meta_path(profile), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load_meta(profile):
    out = {}
    if os.path.exists(meta_path(profile)):
        with open(meta_path(profile), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["rid"]] = r
    return out


def dataset(profile, items=None):
    """[(rid, item, session)] for labelled recordings with session metadata (label text wins over meta item)."""
    labels, meta = profile.labels(), load_meta(profile)
    rows = []
    for rid, m in meta.items():
        if not profile.has(rid):
            continue
        item = labels.get(rid, {}).get("text", m["item"])
        if items is None or item in items:
            rows.append((rid, item, int(m["session"])))
    return sorted(rows, key=lambda r: (r[2], r[0]))


def session_order(items, reps, session, seed=0):
    rng = random.Random(f"{seed}-{session}")
    order = [(it, r) for it in items for r in range(reps)]
    rng.shuffle(order)
    return order


# ---------------------------------------------------------------- preprocessing
def filter_window(x, mains_hz=60, warm=500):
    """Causal filters (as in live use) on one window; reflect-padding warms the filter state."""
    from scipy import signal as sps
    x = np.asarray(x, dtype=np.float64)
    x = x - np.median(x, axis=0, keepdims=True)
    pad = np.concatenate([x[warm:0:-1], x]) if x.shape[0] > warm else x
    y = sps.sosfilt(design_filters(FS, mains_hz), pad, axis=0)
    return y[pad.shape[0] - x.shape[0]:].astype(np.float32)


def load_windows(profile, rows, length=int(WINDOW_S * FS)):
    """Filtered, session-normalised (per-channel std over the session, no labels used) fixed-length windows."""
    mains = profile.settings()["mains_hz"]
    xs = []
    for rid, _, _ in rows:
        x = filter_window(profile.load_raw(rid), mains)
        if x.shape[0] >= length:
            s = (x.shape[0] - length) // 2
            x = x[s:s + length]
        else:
            left = (length - x.shape[0]) // 2
            x = np.pad(x, ((left, length - x.shape[0] - left), (0, 0)))
        xs.append(x)
    X = np.stack(xs) if xs else np.zeros((0, length, 1), np.float32)
    sess = np.array([r[2] for r in rows])
    for s in np.unique(sess):
        m = sess == s
        sd = X[m].reshape(-1, X.shape[2]).std(0) + 1e-6
        X[m] = np.clip(X[m] / sd, -20, 20)
    return X


def augment(x, rng):
    T, C = x.shape
    if rng.random() < 0.7:
        r = rng.uniform(0.9, 1.1)
        src = np.clip(np.arange(T) * r + (T - T * r) / 2, 0, T - 1)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, T - 1)
        f = (src - i0)[:, None]
        x = x[i0] * (1 - f) + x[i1] * f
    x = x * np.exp(rng.normal(0, 0.2, (1, C)))
    x = x + rng.normal(0, 0.05, x.shape)
    x = np.roll(x, int(rng.integers(-150, 151)), axis=0)
    if C > 2 and rng.random() < 0.2:
        x[:, rng.integers(C)] = 0
    for _ in range(rng.integers(0, 3)):
        w = int(rng.integers(20, 80))
        s = int(rng.integers(0, T - w))
        x[s:s + w] = 0
    return x.astype(np.float32)


# ---------------------------------------------------------------- model
class CommandNet(nn.Module):
    def __init__(self, encoder, n_items):
        super().__init__()
        self.encoder = encoder
        d = encoder.config["d_model"]
        self.att = nn.Linear(d, 1)
        self.cls = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, n_items))

    def forward(self, x):
        h, _ = self.encoder.encode(x)
        w = self.att(h).softmax(1)
        return self.cls((w * h).sum(1)), self.encoder.head(h).log_softmax(-1)


def make_encoder(base, n_channels, device):
    if base and base != "none":
        enc, _ = load_model(base, device)
        enc = adapt_channels(enc, n_channels)
    else:
        enc = EMGCTC(n_channels=n_channels, d_model=128, n_layers=3, n_heads=4, ff=256, dropout=0.2)
    return enc.to(device)


def fit(X, y, texts, n_items, base, device, epochs=60, lr=3e-4, ctc_weight=0.3, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = CommandNet(make_encoder(base, X.shape[2], device), n_items).to(device)
    # new parts (re-initialised input block, pooling, classifier) learn 10x faster than the pretrained encoder
    fresh = [p for n, p in net.named_parameters()
             if n.startswith(("att.", "cls.", "encoder.frontend.0.conv1", "encoder.frontend.0.skip"))]
    ids = {id(p) for p in fresh}
    rest = [p for p in net.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": rest, "lr": lr}, {"params": fresh, "lr": lr * 10}], weight_decay=0.01)
    lr = [lr, lr * 10]
    bs = 32
    steps = epochs * int(np.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    labels = [torch.tensor(encode(t)) for t in texts]
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for s in range(0, len(X), bs):
            b = perm[s:s + bs]
            xb = torch.from_numpy(np.stack([augment(X[i], rng) for i in b])).to(device)
            yb = torch.from_numpy(y[b]).long().to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, lp = net(xb)
            loss = F.cross_entropy(logits.float(), yb, label_smoothing=0.1)
            if ctc_weight:
                tg = [labels[i] for i in b]
                T = lp.shape[1]
                loss = loss + ctc_weight * F.ctc_loss(lp.float().transpose(0, 1), torch.cat(tg).to(device),
                                                      torch.full((len(b),), T, dtype=torch.long),
                                                      torch.tensor([len(t) for t in tg]), zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 20 == 0 or ep == 0:
            log({"epoch": ep + 1, "loss": round(tot / len(X), 4)})
    return net


@torch.no_grad()
def predict(net, X, device, items=None, bs=64):
    """Item scores = classifier log-softmax + CTC log-softmax over the items' spellings (equal weights,
    fixed in advance). Returns (combined, classifier_only, ctc_only)."""
    net.eval()
    cls, ctc = [], []
    tg = [torch.tensor(encode(t)) for t in items] if items else None
    for s in range(0, len(X), bs):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, lp = net(torch.from_numpy(X[s:s + bs]).to(device))
        cls.append(logits.float().log_softmax(-1).cpu().numpy())
        if tg is not None:
            lp = lp.float()
            B, T, _ = lp.shape
            sc = np.zeros((B, len(tg)), np.float32)
            for k, t in enumerate(tg):
                sc[:, k] = -F.ctc_loss(lp.transpose(0, 1), t.repeat(B).to(device), torch.full((B,), T, dtype=torch.long),
                                       torch.full((B,), len(t), dtype=torch.long), reduction="none",
                                       zero_infinity=False).cpu().numpy()
            sc = np.where(np.isfinite(sc), sc, -1e4)
            ctc.append(sc - np.logaddexp.reduce(sc, axis=1, keepdims=True))
    cls = np.concatenate(cls) if cls else np.zeros((0, 1))
    if tg is None:
        return cls, cls, cls
    ctc = np.concatenate(ctc)
    return cls + ctc, cls, ctc


def lda_features(X, bins=10):
    T = X.shape[1]
    e = [np.log(np.sqrt((X[:, i * T // bins:(i + 1) * T // bins] ** 2).mean(1)) + 1e-3) for i in range(bins)]
    d = np.diff(X, axis=1)
    zc = [np.log(np.abs(d[:, i * T // bins:(i + 1) * T // bins]).mean(1) + 1e-3) for i in range(bins)]
    return np.concatenate(e + zc, axis=1)


def lda_predict(Xtr, ytr, Xte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(lda_features(Xtr), ytr)
    return clf.predict_proba(lda_features(Xte))


def acc_ci(correct, n_boot=2000, seed=1234):
    c = np.asarray(correct, float)
    if len(c) == 0:
        return 0.0, [0.0, 0.0]
    rng = np.random.default_rng(seed)
    bs = c[rng.integers(0, len(c), (n_boot, len(c)))].mean(1)
    return float(c.mean()), [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def subset_accuracy(scores, y, m, seed=0):
    """Accuracy when only m items are possible (the page shows m of them), true item always included."""
    rng = np.random.default_rng(seed)
    K = scores.shape[1]
    if m >= K:
        return None
    ok = []
    for s, t in zip(scores, y):
        others = rng.choice([k for k in range(K) if k != t], m - 1, replace=False)
        cand = np.concatenate([[t], others])
        ok.append(cand[np.argmax(s[cand])] == t)
    return acc_ci(ok)


# ---------------------------------------------------------------- train / evaluate
def run_fold(X, y, texts, sess, test_sess, items, base, device, epochs, log=print):
    tr, te = sess != test_sess, sess == test_sess
    net = fit(X[tr], y[tr], [texts[i] for i in np.where(tr)[0]], len(items), base, device, epochs=epochs, log=log)
    sc, cls, ctc = predict(net, X[te], device, items)
    lda = lda_predict(X[tr], y[tr], X[te])
    res = {"test_session": int(test_sess), "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "pred": sc.argmax(1).tolist(), "true": y[te].tolist(), "scores": sc, "lda_pred": lda.argmax(1).tolist(),
           "cls_pred": cls.argmax(1).tolist(), "ctc_pred": ctc.argmax(1).tolist()}
    return res, net


def summarise(folds, items):
    true = np.concatenate([f["true"] for f in folds])
    pred = np.concatenate([f["pred"] for f in folds])
    lda = np.concatenate([f["lda_pred"] for f in folds])
    cls_acc = acc_ci(np.concatenate([f["cls_pred"] for f in folds]) == true)
    ctc_acc = acc_ci(np.concatenate([f["ctc_pred"] for f in folds]) == true)
    scores = np.concatenate([f["scores"] for f in folds])
    acc, ci = acc_ci(pred == true)
    lacc, lci = acc_ci(lda == true)
    K = len(items)
    conf = np.zeros((K, K), int)
    for t, p in zip(true, pred):
        conf[t, p] += 1
    pairs = sorted(((int(conf[i, j]), items[i], items[j]) for i in range(K) for j in range(K) if i != j and conf[i, j]),
                   reverse=True)[:10]
    out = {"accuracy": acc, "accuracy_ci95": ci, "n": int(len(true)), "chance": 1.0 / K,
           "lda_accuracy": lacc, "lda_accuracy_ci95": lci,
           "classifier_only_accuracy": cls_acc[0], "ctc_only_accuracy": ctc_acc[0],
           "per_item": {items[k]: float((pred[true == k] == k).mean()) if (true == k).any() else None for k in range(K)},
           "top_confusions": [{"count": c, "said": a, "decoded": b} for c, a, b in pairs]}
    for m in (10, 5):
        r = subset_accuracy(scores, true, m)
        if r:
            out[f"accuracy_given_{m}_on_page"] = {"accuracy": r[0], "accuracy_ci95": r[1]}
    return out


def prepare(user, items=None):
    profile = Profile(user)
    rows = dataset(profile, set(items) if items else None)
    if not rows:
        raise SystemExit(f"profile '{user}' has no command recordings with session metadata "
                         f"(record with `python -m semg.commands record` or tag Nullpoint recordings with `tag`)")
    items = sorted({r[1] for r in rows}) if not items else [i for i in items if any(r[1] == i for r in rows)]
    idx = {it: k for k, it in enumerate(items)}
    rows = [r for r in rows if r[1] in idx]
    X = load_windows(profile, rows)
    y = np.array([idx[r[1]] for r in rows])
    sess = np.array([r[2] for r in rows])
    return profile, rows, items, X, y, sess


def cmd_train(args):
    profile, rows, items, X, y, sess = prepare(args.user, args.items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sessions = sorted(set(sess.tolist()))
    test = args.test_session if args.test_session is not None else sessions[-1]
    texts = [r[1] for r in rows]
    fold, net = run_fold(X, y, texts, sess, test, items, args.base, device, args.epochs)
    summ = summarise([fold], items)
    out = args.out or os.path.join("runs", f"cmd-{args.user}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    torch.save({"model": net.state_dict(), "encoder_config": net.encoder.config, "items": items,
                "base": args.base}, os.path.join(out, "model.pt"))
    _report(out, "train", args, items, sessions, {"official": summ}, [fold])
    print(f"session {test} held out: accuracy {100*summ['accuracy']:.1f}% [{100*summ['accuracy_ci95'][0]:.1f}, "
          f"{100*summ['accuracy_ci95'][1]:.1f}]  (LDA baseline {100*summ['lda_accuracy']:.1f}%, chance {100/len(items):.1f}%)")
    return summ


def cmd_eval(args):
    profile, rows, items, X, y, sess = prepare(args.user, args.items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sessions = sorted(set(sess.tolist()))
    if len(sessions) < 2:
        raise SystemExit("need at least two sessions for a session-held-out evaluation")
    texts = [r[1] for r in rows]
    folds = []
    for s in sessions:
        f, _ = run_fold(X, y, texts, sess, s, items, args.base, device, args.epochs, log=lambda r: None)
        folds.append(f)
        print(f"  held-out session {s}: {100*np.mean(np.array(f['pred']) == np.array(f['true'])):.1f}% "
              f"(n={f['n_test']}, LDA {100*np.mean(np.array(f['lda_pred']) == np.array(f['true'])):.1f}%)", flush=True)
    res = {"official": summarise([folds[-1]], items), "loso": summarise(folds, items)}
    out = args.out or os.path.join("runs", f"cmd-{args.user}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    _report(out, "eval", args, items, sessions, res, folds)
    for k, v in res.items():
        print(f"{k:8s} accuracy {100*v['accuracy']:.1f}% [{100*v['accuracy_ci95'][0]:.1f}, {100*v['accuracy_ci95'][1]:.1f}] "
              f"n={v['n']}  (classifier {100*v['classifier_only_accuracy']:.1f}%, CTC {100*v['ctc_only_accuracy']:.1f}%)  "
              f"LDA {100*v['lda_accuracy']:.1f}%  chance {100*v['chance']:.1f}%")
    return res


def _report(out, mode, args, items, sessions, res, folds):
    rep = {"mode": mode, "user": args.user, "base": args.base, "epochs": args.epochs, "items": items,
           "sessions": sessions, "results": res,
           "folds": [{k: v for k, v in f.items() if k != "scores"} for f in folds],
           "note": "official = last session held out (commands.json protocol); loso = every session held out once"}
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    log_run("commands", os.path.basename(out), user=args.user, mode=mode, base=args.base, n_items=len(items),
            sessions=sessions, metrics={k: {"accuracy": v["accuracy"], "accuracy_ci95": v["accuracy_ci95"],
                                            "lda_accuracy": v["lda_accuracy"], "n": v["n"]} for k, v in res.items()},
            report=os.path.join(out, "report.json"), notes=args.notes)


# ---------------------------------------------------------------- recording
def serial_samples(port, baud=921600, channels=4):
    """Yield (channels,) sample rows from the ESP32 firmware's CSV stream."""
    import serial  # pyserial
    with serial.Serial(port, baud, timeout=1) as s:
        s.reset_input_buffer()
        while True:
            line = s.readline().decode("ascii", "ignore").strip()
            parts = line.split(",")
            if len(parts) == channels:
                try:
                    yield [float(p) for p in parts]
                except ValueError:
                    continue


def simulated_samples(channels=4, seed=0):
    rng = np.random.default_rng(seed)
    while True:
        yield list(2048 + rng.normal(0, 30, channels))


def capture(stream, n):
    return np.array([next(stream) for _ in range(n)], dtype=np.float32)


def cmd_record(args):
    items = args.items or starter_items(set_name=getattr(args, "set", "tier1_starter"))
    profile = Profile(args.user)
    stream = simulated_samples(args.channels) if args.simulate else serial_samples(args.port, channels=args.channels)
    order = session_order(items, args.reps, args.session, args.seed)
    print(f"session {args.session}: {len(order)} prompts ({len(items)} items x {args.reps}). Mouth each item silently "
          f"when it says GO. Ctrl+C stops; finished prompts are saved.")
    done = {(m["item"], m.get("rep")) for m in load_meta(profile).values() if m["session"] == args.session}
    for k, (item, rep) in enumerate(order):
        if (item, rep) in done:
            continue
        print(f"[{k+1}/{len(order)}]  {item.upper()}", flush=True)
        if not args.simulate:
            capture(stream, int(1.0 * FS))            # 1 s countdown (discarded; keeps filter/ADC warm)
        print("        GO", flush=True)
        x = capture(stream, int(WINDOW_S * FS))
        rid = profile.add_recording(x)
        profile.add_label(rid, item, source="prompted")
        add_meta(profile, rid, item, args.session, rep)
    print("saved:", profile.summary())


def cmd_tag(args):
    """Attach a session number to labelled Nullpoint recordings made in a time window."""
    profile = Profile(args.user)
    meta = load_meta(profile)
    items = set(args.items or starter_items(set_name=getattr(args, "set", "tier1_starter")))
    n = 0
    for rid, lab in profile.labels().items():
        if rid in meta or lab["text"] not in items:
            continue
        t = lab["time"]
        if (args.since and t < args.since) or (args.until and t > args.until):
            continue
        add_meta(profile, rid, lab["text"], args.session)
        n += 1
    print(f"tagged {n} recordings as session {args.session}")


# ---------------------------------------------------------------- readiness data
def cmd_synth(args):
    """Synthetic 4-channel recordings: each item is a sequence of phone-like bursts; each session has its own
    electrode gains and offset (so session-held-out evaluation is meaningful)."""
    from .phonemes import TOKEN_TO_ID, pronounce
    items = args.items or starter_items(set_name=getattr(args, "set", "tier1_starter"))
    profile = Profile(args.user)
    rng = np.random.default_rng(args.seed)
    C = args.channels
    pat = rng.uniform(0.1, 1.0, (60, C))
    for s in range(1, args.sessions + 1):
        gain = np.exp(rng.normal(0, 0.3, C))
        for item, rep in session_order(items, args.reps, s, args.seed):
            T = int(WINDOW_S * FS)
            x = rng.normal(0, 0.05, (T, C))
            t = int(rng.integers(200, 700))
            for w in item.split():
                for p in pronounce(w):
                    n = int(rng.integers(70, 110))
                    if t + n >= T:
                        break
                    x[t:t + n] += rng.normal(0, 1, (n, C)) * pat[TOKEN_TO_ID[p]] * np.hanning(n)[:, None]
                    t += n
                t += 60
            x = (x * gain * 200 + 2048 + rng.normal(0, 20, C)).astype(np.float32)
            rid = profile.add_recording(x)
            profile.add_label(rid, item, source="prompted")
            add_meta(profile, rid, item, s, rep)
    print("synthetic profile:", profile.summary())


def cmd_import_gaddy(args):
    """Cut 2.5 s windows around command words in Gaddy silent TRAIN/DEV recordings (never test), keep 4 of 8
    channels, and group the 8 recording days into 3 pseudo-sessions. Readiness test only: the base model was
    trained on these recordings, so accuracy here is not a result."""
    from .data import EMGDataset, build_cache, load_index
    from .isoword import word_spans
    from .text import CHAR_TO_ID, SPACE
    from .train import run_model
    cmds = load_commands()
    vocab = {p for g in cmds["groups"].values() for p in g if " " not in p}
    utts = [u for u in load_index(args.index) if u.silent and u.split in ("train", "dev")]
    utts = [u for u in utts if any(w in vocab for w in u.text.split())]
    build_cache(utts, args.cache, workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.base, device)
    lps = run_model(model, EMGDataset(utts, args.cache), device)
    days = sorted({os.path.basename(os.path.dirname(u.path)) for u in utts})
    day_session = {d: 1 + i % 3 for i, d in enumerate(days)}
    cand = []
    for u, lp in zip(utts, lps):
        ws = u.text.split()
        seq = []
        for i, w in enumerate(ws):
            if i:
                seq.append(SPACE)
            seq.extend(CHAR_TO_ID[c] for c in w)
        spans = word_spans(lp, seq, SPACE)
        if spans is None:
            continue
        for w, (s, e) in zip(ws, spans):
            if w in vocab:
                cand.append((w, u, (s * 8, e * 8)))       # word span in samples (8 samples per frame)
    counts = {}
    for w, u, _ in cand:
        counts.setdefault(w, set()).add(day_session[os.path.basename(os.path.dirname(u.path))])
    per = {}
    for w, _, _ in cand:
        per[w] = per.get(w, 0) + 1
    keep = sorted(w for w in per if per[w] >= args.min_count and len(counts[w]) == 3)
    profile = Profile(args.user)
    n, pad = 0, 150
    rng = random.Random(0)
    rng.shuffle(cand)
    taken = {}
    for w, u, c in cand:
        if w not in keep:
            continue
        s = day_session[os.path.basename(os.path.dirname(u.path))]
        if taken.get((w, s), 0) >= args.max_per_session:
            continue
        raw = np.load(u.path + "_emg.npy")[:, :args.channels]
        a, b = max(0, c[0] - pad), min(raw.shape[0], c[1] + pad)
        if b - a < 200:
            continue
        x = raw[a:b]
        rid = profile.add_recording(x)
        profile.add_label(rid, w, source="prompted")
        add_meta(profile, rid, w, s, taken.get((w, s), 0))
        taken[(w, s)] = taken.get((w, s), 0) + 1
        n += 1
    print(f"imported {n} word windows, items: {keep}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, user=True):
        if user:
            p.add_argument("--user", required=True)
        p.add_argument("--items", nargs="*", default=None, help="restrict to these items (default: all / starter set)")
        p.add_argument("--set", default="tier1_starter", choices=ITEM_SETS,
                       help="item set from vocab/commands.json: tier1_starter, or intent_v1 (verbs + digits)")
        p.add_argument("--notes", default="")

    p = sub.add_parser("plan")
    p.add_argument("--set", default="tier1_starter", choices=ITEM_SETS)
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--base", default="runs/base/best.pt", help="pretrained checkpoint, or 'none' to train from scratch")
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--out", default=None)
        if name == "train":
            p.add_argument("--test-session", type=int, default=None, help="default: the last session")
    p = sub.add_parser("record")
    common(p)
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--port", default=None, help="serial port of the ESP32, e.g. COM5 (needs pyserial)")
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--simulate", action="store_true", help="no hardware: record simulated noise (tests the flow)")
    p = sub.add_parser("tag")
    common(p)
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--since", default=None, help="ISO time, compared with the label time")
    p.add_argument("--until", default=None)
    p = sub.add_parser("synth")
    common(p)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("import-gaddy")
    common(p)
    p.add_argument("--base", default="runs/base/best.pt")
    p.add_argument("--index", default="data/index.json")
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--min-count", type=int, default=24)
    p.add_argument("--max-per-session", type=int, default=12)
    args = ap.parse_args()

    if args.cmd == "plan":
        c = load_commands()
        st = c[args.set]
        items = st["items"]
        print(f"{args.set} ({len(items)} items): {', '.join(items)}\n\nWhy: {st['why']}\n")
        n = len(items) * args.reps
        print(f"Session {args.session}: {n} prompts, about {n * (1 + WINDOW_S + 0.5) / 60:.0f} min. Order:")
        print(", ".join(it for it, _ in session_order(items, args.reps, args.session, args.seed)[:40]), "...")
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "record":
        if not args.simulate and not args.port:
            raise SystemExit("give --port COMx (pip install pyserial), or --simulate")
        cmd_record(args)
    elif args.cmd == "tag":
        cmd_tag(args)
    elif args.cmd == "synth":
        cmd_synth(args)
    elif args.cmd == "import-gaddy":
        cmd_import_gaddy(args)


if __name__ == "__main__":
    main()
