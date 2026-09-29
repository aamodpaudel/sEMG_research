# Afono evaluation protocol and metrics

Protocol **v2** (2026-09-29). Implemented in `semg/evaluate.py`, `semg/metrics.py` and `semg/compare.py`.
The website's research card (`site/research.html`) is generated from the same outputs.

## What each run produces
| File | Contents |
|---|---|
| `runs/<run>/report.json` | Every task: WER, CER, word accuracy, sentence accuracy, 95% CIs, speed, ITR; dataset split hashes; decoder settings; compute and versions |
| `runs/<run>/predictions.jsonl` | One line per test utterance: task, id, duration, reference, greedy output, final output |
| `runs/registry.jsonl` | One line per train / evaluate / calibrate / compare run, with notes and git revision |

## Data and splits
- Gaddy & Klein (2020) silent-speech EMG, DOI 10.5281/zenodo.4064408. One speaker, 8 channels, 1,000 Hz.
- Book sentences: official `testset_largedev.json` (199 dev, 98 test silent utterances in our index). Voiced recordings of dev/test sentences are excluded from training.
- Dates and times: no official split. 10% dev and 10% test of the silent recordings, chosen by a hash of the file path (42 dev, 43 test).
- Split hashes (SHA-256 of sorted utterance ids) are stored in each report so a changed split is detectable.

## Tasks
| Task | Test data | Lexicon |
|---|---|---|
| `dates` | silent dates-and-times test | the 51 words in its training transcripts |
| `closedN` (N = 100, 250, 500) | silent book test sentences, kept in index order while their combined vocabulary fits N words | exactly N words: the test sentences' words plus the most frequent training words |
| `open` | all 98 silent book test sentences | all 9,139 training words; out-of-vocabulary test words (4.1%) count as errors |

Decoder hyperparameters (prior weight, word bonus) are tuned on the first 100 dev utterances only.
Protocol v1 pooled dev and test sentences for closed-N; v2 fixed that, and the 500-word result moved from 45.2% to 39.7%.

## Metrics
- **WER**: corpus-level edit distance over words divided by reference words. **Word accuracy** = 1 − WER. **CER** likewise over characters. Text is lowercased, numbers spelled out, punctuation removed.
- **Sentence accuracy**: share of utterances decoded exactly.
- **95% CI**: percentile bootstrap over utterances, 2,000 resamples, seed 1234.
- **Paired comparison** (`semg.compare`, and automatically after every calibration): paired bootstrap of WER(B) − WER(A) on identical utterances; reports the 95% CI and the one-sided p of "B is not better".
- **ITR** (Wolpaw et al., 2002), one selection = one word from an N-word vocabulary:
  `B = log2 N + P log2 P + (1 − P) log2((1 − P)/(N − 1))` bits per word; ITR = B × words per minute.
  It assumes equiprobable words and uniform errors, which sentences violate, and it favours large N at low accuracy. Reported for comparison with BCI literature only.
- **Correct words per minute** = reference words ÷ recording minutes × word accuracy (natural reading pace, no interface overhead).
- **Real-time factor** = (median encoder time + decoder time) ÷ recording duration.

## Results, run `base`, protocol v2
| Task | Vocab | Utterances / words | Word accuracy [95% CI] | No lexicon | Correct wpm |
|---|---|---|---|---|---|
| dates | 51 | 43 / 172 | 94.8% [90.9, 98.2] | 57.6% | 78.7 |
| closed100 | 100 | 11 / 150 | 37.3% [20.9, 51.7] | 23.3% | 52.8 |
| closed250 | 250 | 25 / 438 | 48.9% [42.0, 55.3] | 24.4% | 72.5 |
| closed500 | 500 | 64 / 1,086 | 39.7% [33.9, 45.2] | 24.7% | 55.9 |
| open | 9,139 | 98 / 1,643 | 40.5% [36.8, 44.1] | 25.6% | 58.2 |

Calibration on a demo profile (155 labelled recordings, same speaker): held-out WER 60.6% → 58.9%, change −1.7 points, paired 95% CI [−6.2, +2.9], p = 0.25, n = 44. **Not significant.**

Published open-vocabulary silent WER on this dataset: previous state of the art 28.8%, MONA LISA 12.2% (Benster et al., 2024). Ours: 59.5%.

## Decoder ablation (run `base`, protocol v2)
A bigram word prior (interpolated absolute discounting, D = 0.75, training transcripts only) against the unigram prior,
same model and test utterances, paired bootstrap:

| Task | WER unigram | WER bigram | Change [95% CI] | p |
|---|---|---|---|---|
| closed500 | 60.3% | 53.5% | −6.8 [−8.8, −4.8] | < 0.001 |
| open | 59.5% | 56.8% | −2.7 [−5.0, −0.5] | 0.006 |
| closed100 | 62.7% | 46.7% | −16.0 [−29.0, −4.9] | 0.002 |
| closed250 | 51.1% | 47.3% | −3.9 [−7.5, 0.0] | 0.033 |
| dates | 5.2% | 4.1% | −1.2 [−3.0, 0.0] | 0.125 (not significant) |

The bigram decoder is now the default in Nullpoint. The browser demo still uses the unigram prior.

## Afono DOM Model: DOM-0 fixture
`python -m semg.dom_eval` runs the detector in headless Chrome on `extension/test/fixture.html`: 17 true targets
(links, buttons, an icon button with aria-label, ARIA roles, two script-only click handlers, form fields, a checkbox,
an image link, a summary) and 6 decoys (disabled, display:none, visibility:hidden, aria-hidden, covered, off-screen).
v0 result: precision 100%, recall 100%, label accuracy 100%, 5 of 5 commands parsed. We wrote this page ourselves,
so it is a regression test, not evidence of real-world performance.

## Afono Track Model: session protocol
Accuracy = distance between each validation target and the median estimate over 1 s (after 0.6 s settling), averaged
over 5 targets not used in calibration. Precision = RMS distance between consecutive estimates. Both in pixels and
degrees of visual angle from the entered screen width and viewing distance. Target test: 12 cells in a 4 × 3 grid,
random order, selection by Space when the gaze-snapped highlight is on the goal; reports hit rate and time per selection.

## Planned protocols for the other two models
- **Afono Track Model** (gaze): angular error (degrees) on MPIIFaceGaze, leave-one-subject-out; on-screen pointing hit rate and error in pixels at a fixed viewing distance.
- **Afono DOM Model** (page elements): precision and recall of actionable elements on the WebUI dataset; median size of the per-page vocabulary handed to the decoder.
- **Combined system**: success rate and time per task on scripted web tasks, selections per minute and ITR, against keyboard, mouse and eye-gaze baselines. Participants with and without motor impairments, with ethics review before any human study.

## Rules
1. Change one thing per run and say what in `--notes`.
2. Never tune on test. A protocol change gets a new version number here and in `semg/evaluate.py`.
3. Report every run, including the ones that got worse.
