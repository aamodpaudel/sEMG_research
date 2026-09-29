"""Package a checkpoint for the in-browser demo (site/demo/).

    python -m semg.web_export --ckpt runs/base/best.pt

Writes model.onnx (int8 matmuls, fp32 convs), samples.bin/.json (the FIRST test
recordings in index order, not hand-picked), lexicons and python reference outputs
that the JS decoder is checked against.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .data import load_index
from .decode import Lexicon, greedy, lexicon_beam_search
from .export import _Wrapper
from .model import load_model
from .text import CHARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="site/demo")
    ap.add_argument("--n-closed", type=int, default=6)
    ap.add_argument("--n-open", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, ckpt = load_model(args.ckpt, "cpu")
    fp32 = os.path.join(args.out, "_fp32.onnx")
    dummy = torch.randn(1, 4000, model.config["n_channels"])
    torch.onnx.export(_Wrapper(model).eval(), (dummy,), fp32, input_names=["emg"], output_names=["logprobs"],
                      dynamic_axes={"emg": {1: "time"}, "logprobs": {1: "frames"}}, opset_version=17, dynamo=False)
    q = os.path.join(args.out, "model.onnx")
    quantize_dynamic(fp32, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    os.remove(fp32)

    utts = load_index(args.index)
    closed = [u for u in utts if u.split == "test" and u.silent and u.closed_vocab][: args.n_closed]
    opened = [u for u in utts if u.split == "test" and u.silent and not u.closed_vocab][: args.n_open]
    train = [u for u in utts if u.split == "train"]
    dates_counts = Counter(w for u in train if u.closed_vocab for w in u.text.split())
    open_counts = Counter(w for u in train if not u.closed_vocab for w in u.text.split())
    lex = {"dates": Lexicon(dates_counts.keys(), dates_counts), "open": Lexicon(open_counts.keys(), open_counts)}

    sess = ort.InferenceSession(q, providers=["CPUExecutionProvider"])
    meta, chunks, offset, agree = [], [], 0, []
    for kind, group in (("dates", closed), ("open", opened)):
        for u in group:
            x = np.load(os.path.join(args.cache, u.uid + ".npy")).astype(np.float32)
            with torch.no_grad():
                ref = model(torch.from_numpy(x)[None])[0][0].numpy()
            lp = sess.run(None, {"emg": x[None]})[0][0]
            agree.append(float((lp.argmax(-1) == ref.argmax(-1)).mean()))
            meta.append({"id": u.uid, "kind": kind, "text": u.text, "offset": offset, "length": int(x.shape[0]),
                         "channels": int(x.shape[1]), "py_greedy": greedy(lp),
                         "py_lexicon": lexicon_beam_search(lp, lex[kind], beam=32, lm_weight=1.0, word_bonus=3.0)})
            chunks.append(x.reshape(-1))
            offset += x.size
    np.concatenate(chunks).astype("<f4").tofile(os.path.join(args.out, "samples.bin"))
    info = {"model": os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))), "epoch": ckpt.get("epoch"),
            "chars": CHARS, "blank": 0, "sample_rate": 1000, "frame_rate": 125,
            "decoder": {"beam": 32, "lm_weight": 1.0, "word_bonus": 3.0},
            "int8_argmax_agreement": round(float(np.mean(agree)), 4), "samples": meta}
    with open(os.path.join(args.out, "samples.json"), "w") as f:
        json.dump(info, f)
    with open(os.path.join(args.out, "lexicons.json"), "w") as f:
        json.dump({k: dict(c) for k, c in (("dates", dates_counts), ("open", open_counts))}, f)
    size = os.path.getsize(q) / 1e6
    print(f"model {size:.1f} MB, int8 argmax agreement {np.mean(agree):.4f}, samples {len(meta)}, "
          f"samples.bin {os.path.getsize(os.path.join(args.out, 'samples.bin')) / 1e6:.1f} MB")
    for m in meta:
        print(m["kind"], "|", m["text"], "->", m["py_lexicon"])


if __name__ == "__main__":
    main()
