# Afono: sEMG silent speech → text

| Part | Where | Run it |
|---|---|---|
| Model + training | `semg/` | `python -m semg.train`, `python -m semg.evaluate` |
| Learning loop (per-user calibration) | `semg/user.py`, `semg/calibrate.py` | `python -m semg.calibrate --profile <name>` |
| Afono Lab, the local app | `app/` | `python -m app.server`, then open http://127.0.0.1:8765 |
| Website with in-browser demo | `site/` | `python -m http.server 8000 -d site`, then open http://127.0.0.1:8000 |
| Run registry (every experiment) | `runs/registry.jsonl` | written automatically |
| Research metrics, paired tests | `semg/metrics.py`, `semg/compare.py`, [METRICS.md](METRICS.md) | `python -m semg.compare --profile P --a A.pt --b B.pt` |
| Brand | `BRAND.md`, `site/brand.html` | |
| $50 rig firmware (untested) | `firmware/esp32_emg/` | Arduino IDE, ESP32 board |

## The loop we follow for every experiment
1. **Train or calibrate.** `python -m semg.train --out runs/<name> --notes "what this tests"` or Calibrate in Afono Lab.
   Each run appends one line to `runs/registry.jsonl`.
2. **Benchmark.** `python -m semg.evaluate --ckpt runs/<name>/best.pt --out runs/<name>/report.json`.
   This also rebuilds the website's numbers (`site/data/progress.*`).
3. **Ship the demo model** when a run beats the current one: `python -m semg.web_export --ckpt runs/<name>/best.pt`.
4. **Update the roadmap text** in `site/data/roadmap.json` when a phase changes, then `python -m semg.site_data`.
5. **Publish the site** (see PLAN.md, "How we proceed").

The website never contains a number that is not in the registry.

## Afono Lab: teaching the model
- Pick a recording, choose a vocabulary, press Decode.
- "Words on screen" limits the decoder to the words you type: this is how page-aware decoding will work.
- For your own recordings, type what you actually said and press Save correction. The Personal vocabulary uses it at once.
- "Add demo recordings" fills a profile with real labelled dataset recordings, so calibration can be tried before hardware exists.
- Calibrate fine-tunes on the profile's labelled recordings, mixing in original training data so the model does not forget.
  It measures held-out recordings before and after and logs both. A rig with a different number of electrodes is handled
  by re-initialising only the first layer.

This is supervised learning from corrections. Reward-based learning (reinforcement from accept/undo signals) needs real
usage data first; it is planned for Phase 3, once the browser extension produces those signals.

Phase 1 of [PLAN.md](PLAN.md): an EMG-to-text model trained on the public
Gaddy & Klein silent-speech dataset, with a decoder that can be limited to a
small vocabulary (500 words, or "the buttons on this web page").

## Layout
| Path | What it does |
|---|---|
| `semg/signal.py` | Causal filtering (drift high-pass, mains notch, low-pass) + augmentation |
| `semg/data.py` | Dataset indexing with the official split, preprocessed cache, length-bucketed batches |
| `semg/model.py` | Conv front-end (1000 Hz → 125 Hz) + Transformer, character CTC output (~5.8 M params) |
| `semg/decode.py` | Greedy CTC and lexicon-constrained beam search with a word prior |
| `semg/train.py` | Training loop (bf16 autocast, cosine LR, dev WER checkpointing) |
| `semg/evaluate.py` | Benchmark: open-vocab WER, Closed-N word accuracy, dates/times subset |
| `semg/export.py` | ONNX fp32 + int8 export for the browser, with a parity check |
| `tests/` | Synthetic end-to-end test (no download needed) |

## Run
```bash
pip install -r requirements.txt
python -m pytest -q                        # ~90 s synthetic smoke test

# data: ~3.9 GB download, see PLAN.md for the source
curl -L -o data/raw/emg_data.tar.gz https://zenodo.org/api/records/4064409/files/emg_data.tar.gz/content
tar -xzf data/raw/emg_data.tar.gz -C data/raw
curl -o data/testset_largedev.json https://raw.githubusercontent.com/dgaddy/silent_speech/main/testset_largedev.json

python -m semg.train --out runs/base       # builds index + cache on first run
python -m semg.evaluate --ckpt runs/base/best.pt --out runs/base/report.json
python -m semg.export --ckpt runs/base/best.pt --out runs/base/model.onnx
```

## How to read the benchmark
- **open_greedy / open_lexicon WER**: any sentence from the books. The hard setting, comparable to papers.
- **closed_n[500].word_accuracy**: test sentences whose words fit in a 500-word vocabulary,
  decoded with exactly that vocabulary. This is the product target (≥ 90%).
- **closed_vocab_dates**: the dataset's own dates-and-times subset.

The dataset has one speaker and lab hardware. These numbers validate the method, not the
accuracy a new user on a $50 rig will get; that needs per-user calibration (Phase 4).

## Baseline results (run `base`, protocol v2)
Silent speech only, 95% bootstrap CIs. Full protocol and definitions: [METRICS.md](METRICS.md).

| Task | Vocabulary | Word accuracy [95% CI] | Without the vocabulary |
|---|---|---|---|
| Dates and times | 51 words | **94.8%** [90.9, 98.2] | 57.6% |
| Book sentences | 500 words | 39.7% [33.9, 45.2] | 24.7% |
| Any sentence | 9,139 words | 40.5% [36.8, 44.1] | 25.6% |

The published state of the art on open vocabulary is 12.2% WER (87.8% accuracy); this baseline is at 59.5% WER.

Export (`semg.export`), 5 s window on CPU:

| File | Size | Latency | Matches PyTorch |
|---|---|---|---|
| model.onnx (fp32) | 23 MB | 49 ms | yes |
| model.int8.onnx | 6 MB | 249 ms | same argmax, slower on CPU |
