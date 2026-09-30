# sEMG Silent-Speech + Browser Interface — Build Plan

Goal: let people control a computer and write text without a mouse or keyboard,
using cheap non-invasive sensors, starting with people who have disabilities.

Guiding rule: **small vocabulary, high accuracy first.** 90% on 500 words beats
30% on 50,000 words, because only the first one is usable.

---

## 0. Framing that keeps us honest (and fundable)

| Claim | Reality | How we pitch it |
|---|---|---|
| "Brain to text" | sEMG reads **muscle** activation, not the brain. Non-invasive EEG text decoding is far from usable today. | "Neuromotor interface": we decode the motor commands the brain sends to speech and hand muscles. |
| Public datasets → product | Public EMG datasets are single-speaker or use lab hardware. Weights **will not transfer** directly to a $50 rig or a new face. | Public data validates the architecture and gives pretraining. The product metric is accuracy **after N minutes of per-user calibration on our hardware**. |
| Hardware is the moat | Cheap EMG boards are commodity. | The moat is (a) context-constrained decoding from the DOM, (b) multimodal fusion, (c) per-user adaptation data. |

### Where defensibility actually comes from
1. **DOM-constrained decoding.** The browser knows what is on screen: link texts,
   button labels, form fields. If a page has 14 clickable things, the decoder picks
   1 of 14, not 1 of 500. This turns a mediocre signal into a reliable command.
2. **Multimodal fusion.** Gaze picks *where*, a muscle twitch or a jaw clench confirms
   *click*, and silent speech supplies *words*. Each modality covers another's weak spot.
3. **Per-user adaptation loop.** Every correction becomes labeled data for that user.
   Nobody else has it.
4. **On-device and private.** The model runs in the browser (ONNX Runtime Web / WebGPU).
   No audio or biosignals leave the device, which matters for medical buyers.

### Who to serve first (disability focus)
Silent speech needs working lips, tongue and jaw. That points to:
- **Laryngectomy / voice loss** (articulators intact, voice gone): best fit for silent speech.
- **Spinal cord injury / tetraplegia** (face intact, hands not): silent speech + gaze for full computer control.
- **Late-stage ALS**: articulators often fail too. Gaze plus a single residual muscle switch is the realistic channel.

---

## 1. Datasets

| Dataset | What it is | Use it for |
|---|---|---|
| **Gaddy & Klein 2020, "Silent Speech EMG"** (Zenodo 4064409, ~20 h, 8 facial channels, 1 speaker) | Silent + voiced reading, plus a closed-vocab (dates/times) subset | **Phase 1 benchmark.** Main dataset. |
| **EMG-UKA Trial Corpus** (Karlsruhe, CSL) | Older facial EMG speech corpus, several speakers | Cross-speaker sanity check |
| **emg2qwerty** (Meta, 108 users, 346 h wrist sEMG while typing) | Keystroke decoding from wrist | Phase 5 "typing without a keyboard" |
| **generic-neuromotor-interface** (Meta, 100 users: gestures, handwriting, wrist) | Wrist sEMG gestures & handwriting | Click/scroll gestures, handwriting input |
| **NinaPro DB1–DB10, GRABMyo (PhysioNet), putEMG** | Forearm gesture sEMG | Pretraining a 1–2 channel "switch/click" detector |
| **MPIIFaceGaze, GazeCapture** | Webcam gaze | Only if off-the-shelf MediaPipe/WebGazer gaze is not accurate enough |

---

## 2. Phases, each with an exit criterion

### Phase 1: Offline benchmark on Gaddy data (weeks 1–4) ← **we are here**
**Status (2026-09-29):** pipeline built, baseline trained. Dates/times (51 words) = 94.8% silent word accuracy.
Closed-500 book sentences = 39.7% [33.9, 45.2] under protocol v2 (45% under v1, which pooled dev into test). Open vocab = 59.5% WER. See README for the table.
Next iterations, in order of expected gain:
1. Word bigram/trigram prior in the lexicon decoder (book sentences have no structure for a unigram prior).
2. Phoneme CTC targets plus a pronunciation lexicon, as in Gaddy 2021.
3. Transfer voiced audio-aligned targets onto silent EMG (DTW alignment) to use the parallel data.
4. Stronger regularisation (train loss 0.39 while dev stalled after epoch ~55).
- Char-level CTC model (conv front-end + small Transformer), trained on silent and voiced EMG.
- Lexicon-constrained beam search: the decoder may only emit words from a chosen vocabulary.
- Report three numbers on the official silent test split:
  1. Open-vocabulary WER (compare with published: Gaddy 2021 ≈ 28–36% WER, Benster 2024 ≈ 12% with LLM rescoring).
  2. **Closed-500 word accuracy**: vocabulary = 500 most frequent words, on test sentences fully inside it.
  3. Closed-vocab (dates/times) subset WER.
- **Exit:** Closed-500 word accuracy ≥ 90% silent. If we miss, shrink to 200 words and add a word-bigram prior before moving on.

### Phase 2: Make it tiny and fast (weeks 4–6)
- Export to ONNX, int8 quantize, target < 5 MB and < 30 ms per 1 s window in the browser.
- Streaming inference: causal filters (already used), chunked encoder, running normalization.
- **Exit:** Browser inference within 2% absolute of PyTorch accuracy.

### Phase 3: Browser layer, Chrome extension first (weeks 5–9)
Do not fork Chromium yet. An MV3 extension gets 90% of the value.
- `dom_targets`: extract visible actionable elements (links, buttons, inputs, ARIA roles) with their labels.
- Command grammar: `click <target>`, `scroll up/down`, `back`, `type <words>`, `select <n>`.
- Gaze: MediaPipe Face Landmarker iris → screen point → highlights nearest DOM target.
- Signal input: Web Serial / Web Bluetooth from the ESP32.
- **Build it "Wizard-of-Oz" first**: fake decoder output with the keyboard to test the UX before hardware exists.
- **Exit:** complete a task (search, open result, fill a form) using only simulated decoder output.

### Phase 4: $50 hardware + own data (weeks 8–14)
Suggested bill of materials (prices approximate):
- 4× Upside Down Labs **BioAmp EXG Pill** (~$10–15 each): chin, under-jaw, cheek, throat.
- 1× **ESP32** dev board (~$5): 12-bit ADC, Bluetooth, 1 kHz sampling.
- Gel electrodes + snap cables (~$10). Battery-powered only (never USB-to-mains while on skin).
- Budget alternative for a single switch: 1× MyoWare 2.0 on the forearm or jaw.
- Collect own dataset with an in-browser recorder: 500 words × 10 reps silent, plus 100 command phrases.
- Fine-tune the Phase 1 model (reinitialise the input conv for the new channel layout).
- **Exit:** ≥ 90% on 500 words for the founder, then 3 other people, after ≤ 20 minutes of calibration.

### Phase 5: Pilots, then general public
- Pilot with laryngectomy and SCI communities and rehab clinicians. Measure words/minute and task completion.
- Wrist sEMG typing/handwriting (emg2qwerty-style) for the general-public version.
- Stay a non-medical accessibility input device at first. Get regulatory advice before any medical claims.

---

## 3. Safety and ethics checklist
- Battery-powered electronics only. Isolation matters more than cost.
- Biosignals are health data. Keep processing on-device and store nothing without consent.
- Involve disabled users in design from Phase 3, not after launch.

---

## How we proceed (updated 2026-09-29)

### What exists now
- Baseline model, benchmark, run registry, per-user calibration loop (tested on real data).
- Afono Nullpoint desktop app, in-browser demo (checked in Chrome: output matches Python exactly), pitch website, brand.

### Next 6 weeks, in order
1. **Put the project in git and on GitHub.** `git init`, commit, push. Commit `runs/registry.jsonl` and the reports, not `data/`
   or checkpoints (upload good checkpoints to Hugging Face instead). Accelerators look for a public repo.
2. **Deploy the site to afono.org.** Point the domain at Cloudflare Pages or GitHub Pages with `site/` as the root.
   Contact on the site: Aamod Paudel, contact@aamodpaudel.com.np (done).
3. **Close the gap on book sentences** (Phase 1 exit test). One change per run, each with `--notes`:
   word-bigram prior in the decoder → phoneme targets → silent/voiced alignment → more regularisation.
4. **Build the Chrome extension** that sends the page's visible link and button labels to the decoder as "Words on screen".
   Test it first with a fake decoder, then with dataset recordings.
5. **Order the hardware** once any Phase 1 task passes 90% on 500 words: ESP32 + 4 BioAmp EXG Pills + gel electrodes.
   Record yourself with Afono Nullpoint (500 words × 10 repetitions), then Calibrate.

### After every run
Evaluate → the site data rebuilds itself → republish the site. The training log section then shows the new run.

### Applying to accelerators
- **Google.org Accelerator: Generative AI** is for nonprofits, civic entities, academic institutions and social enterprises.
  It gives funding, six months of pro bono help from Google employees and Cloud credits. Check each year's focus areas and
  eligibility; most Google.org programs need a registered nonprofit or a fiscal sponsor.
- Register the nonprofit (or join a fiscal sponsor) early. It unlocks Google for Nonprofits, cloud credits and most grants.
- What reviewers look for, and where the site already shows it: a real demo (in-browser model), honest evidence
  (run log, weak results shown), a clear first user (voice loss, spinal cord injury), and a specific ask.
- Biggest gap to fix before applying: evidence from a real person on your own hardware, even one person, even 50 words.

## Positioning (2026-09-29)
- **Organisation:** Afono is a nonprofit AI lab for hands-free computing. One interface, three models: gaze (where), silent speech (what), page understanding (what is possible).
- **Lead with one model.** Silent speech has the evidence; the Track Model and DOM Model are the roadmap, each with a defined benchmark (METRICS.md).
- **Vision line:** "Accessible brain–computer interfaces start with a simple human–computer interface." Use it as the vision, not a product claim: sEMG reads muscles, it is not a BCI.
- **Two audiences:** people who cannot use their hands or voice first; everyone with tired hands second ("use a computer without lifting a finger").
- **Research track:** every run follows METRICS.md, so results can go straight into a workshop paper (e.g. interface-constrained decoding as the contribution).

## Status (2026-09-29, evening)
- **Afono Nullpoint** desktop app built (Windows, `dist/`), runs the three models locally, works without the dataset.
- **sEMG Model:** bigram decoder is significantly better on sentences (500-word 39.7% → 46.5%, p < 0.001).
- **DOM Model v0:** Chrome extension, Tier 1 commands, Nullpoint bridge; passes the DOM-0 fixture.
- **Track Model v0:** webcam gaze with calibration and scored tests; no human session logged yet.
- **Next:** first Track session on yourself; WebUI benchmark for the DOM Model; phoneme-target sEMG run; hardware and Tier 1 recordings.

## Status (2026-09-30, sEMG Model)
- **Protocol v3** (METRICS.md): page-vocabulary tasks (page10/20/30, easy and look-alike "hard" distractors) and an isolated-word proxy, defined before results.
- **Best run `r1-phone`:** phoneme CTC auxiliary head + decoding through the CMUdict pronunciation lexicon. Against `base` (paired, test): 500-word 45.3% → 54.9% (p = 0.001), open WER 58.2% → 52.0% (p < 0.001), page30-hard 60.0% → **68.2% [61.4, 73.7]** (p = 0.02). The 90% page30-hard target is not reached.
- Tried and not kept: trigram Kneser-Ney prior (open −1.5 points, p = 0.04, not better on closed sets); DTW voiced-EMG distillation (page30-hard worse, p < 0.01); stronger augmentation (training stalled, stopped at epoch 26).
- **Tier 1 ready:** `python -m semg.commands plan | record | train | eval` for the 27-word starter set (vocab/commands.json, `tier1_starter`), tested end to end on synthetic 4-channel and Gaddy-derived data.
- **Next:** record Tier 1 sessions 1–3 this weekend and run `semg.commands eval`; that number, not the Gaddy simulation, decides the product claim.
