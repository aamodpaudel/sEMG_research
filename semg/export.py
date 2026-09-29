"""Export a checkpoint to ONNX (fp32 + int8) for in-browser inference with onnxruntime-web.

    python -m semg.export --ckpt runs/base/best.pt --out runs/base/model.onnx
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .model import load_model
from .text import CHARS


class _Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, emg):  # (1, T, C) preprocessed EMG at 1000 Hz, silent mode
        return self.model(emg)[0]


def export(ckpt, out, seconds=5.0):
    model, _ = load_model(ckpt, "cpu")
    n_ch = model.config["n_channels"]
    dummy = torch.randn(1, int(1000 * seconds), n_ch)
    wrapper = _Wrapper(model).eval()
    torch.onnx.export(wrapper, (dummy,), out, input_names=["emg"], output_names=["logprobs"],
                      dynamic_axes={"emg": {1: "time"}, "logprobs": {1: "frames"}}, opset_version=17,
                      dynamo=False)
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_out = out.replace(".onnx", ".int8.onnx")
    quantize_dynamic(out, q_out, weight_type=QuantType.QInt8)
    with torch.no_grad():
        ref = model(dummy)[0].numpy()
    report = {}
    for path in (out, q_out):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"emg": dummy.numpy()})[0]
        t0 = time.perf_counter()
        for _ in range(10):
            sess.run(None, {"emg": dummy.numpy()})
        report[os.path.basename(path)] = {
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "ms_per_%gs_window" % seconds: round((time.perf_counter() - t0) * 100, 1),
            "argmax_agreement": float((y.argmax(-1) == ref.argmax(-1)).mean()),
            "max_abs_logprob_diff": float(np.abs(y - ref).max()),
        }
    with open(os.path.join(os.path.dirname(out) or ".", "vocab.json"), "w") as f:
        json.dump({"chars": CHARS, "blank": 0, "sample_rate": 1000, "frame_rate": 125,
                   "n_channels": n_ch}, f)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--out", default="runs/base/model.onnx")
    args = ap.parse_args()
    print(json.dumps(export(args.ckpt, args.out), indent=2))


if __name__ == "__main__":
    main()
