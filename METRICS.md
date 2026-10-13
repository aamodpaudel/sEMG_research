# Afono evaluation protocol and metrics

Protocol **v3** (2026-09-29; v2 below plus the v3 section). Implemented in `semg/evaluate.py`, `semg/metrics.py` and `semg/compare.py`.
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

## Protocol v3 (2026-09-29): page-vocabulary tasks and an isolated-word proxy
Written down before any v3 result was looked at. Everything in v2 still applies. Changes:

1. **Transcript fix.** Em/en dashes and ellipses now separate words (v2 glued them: "cellar—something" became
   "cellarsomething"). 266 transcripts change, 2 of them in the silent test set. Splits are unchanged (the index
   rebuild asserts identical utterance ids and splits). v2 numbers are therefore re-measured under v3.
2. **Tuning grid** for the sentence tasks: prior weight {0, 0.5, 1, 1.5, 2} × word bonus {0, 1, 2, 3, 4}, first 100 dev utterances.
3. **Word prior options**: unigram, bigram (as v2), and trigram = interpolated modified Kneser-Ney (`semg/lm.py`),
   trained on training transcripts only, no external text. Closed lexicons renormalise the prior over their words.

### Page-vocabulary tasks (`pageN-easy`, `pageN-hard`, N = 10, 20, 30)
Simulates "the page tells the decoder which ~N words are possible" (the product's 20–30 command words).
- Test data: all 98 silent book test sentences. None is dropped.
- Lexicon, per utterance: the utterance's k distinct words plus d = max(N − k, ⌈N/3⌉) distractors from the training
  vocabulary. So the lexicon has exactly N words when k ≤ N − ⌈N/3⌉, and more than N for long sentences (every lexicon
  keeps at least ⌈N/3⌉ wrong choices). The report gives the mean, minimum and maximum lexicon size.
- **easy** distractors: training words ranked by SHA-256 of ("afono-page-v3", utterance id, word).
- **hard** distractors: for each utterance word in order of first appearance, round robin, its next-nearest training
  word by character edit distance (ties by the same hash). These are the look-alike words ("them"/"then"/"thee").
  **page30-hard is the headline** page number.
- Prior: uniform over the page words (a page gives a set of words, not a sentence model). Only the word bonus is tuned,
  grid {−2, −1, 0, 1, 2, 3, 4}, on the first 100 dev utterances with the same construction.
- Secondary (not a headline): `page30-hard+lm`, the same lexicons with the trigram prior renormalised over the page
  words, prior weight {0.5, 1, 1.5} × bonus {−1, 0, 1, 2, 3} tuned on dev.
- Score: word accuracy = 1 − WER of the whole decoded sentence (insertions and deletions count), bootstrap CI over utterances.

### Isolated-word proxy (`isoN-easy`, `isoN-hard`, N = 10, 20, 30; books and dates)
CTC forced alignment of the reference to the model's own output gives each test word a frame span; the span ± 16 ms
is scored against N candidates (the true word + N − 1 distractors chosen as above for that single word) by CTC
log-likelihood; correct if the true word wins. Failed alignments count as errors. It measures word discrimination with
known word boundaries. It is optimistic: the boundaries come from the true word and the encoder hears the whole
sentence. It is not a measurement of isolated words spoken on their own.

### What these numbers are and are not
A **single-speaker, lab-hardware simulation** on Gaddy & Klein book sentences. It is not a measurement on the Afono
4-channel rig, on isolated command words, or on the target user. The Tier 1 command recordings (`python -m semg.commands`)
are the measurement that counts; until they exist, page-task numbers are evidence about the method only.

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

## Results, protocol v3 (2026-09-30)
Silent test word accuracy [95% bootstrap CI]. Sentence tasks use the bigram prior (tuned on dev). Page tasks use a
uniform prior over the page words. "phone" = decoding through the CMUdict pronunciation lexicon with the phoneme head.

| Run | Change | dates | closed500 | open WER | page10 easy / hard | page20 easy / hard | page30 easy / **hard** | iso30 hard |
|---|---|---|---|---|---|---|---|---|
| base (char) | v2 model, re-scored under v3 | 95.9 [92.5, 98.8] | 45.3 [39.4, 51.0] | 58.2 [54.2, 62.1] | 65.6 / 65.2 | 66.0 / 65.6 | 66.5 / **60.0** [54.5, 64.8] | 59.4 |
| r0 (char) | trigram MKN prior instead of bigram | 97.1 | 45.7 | 56.7 | same (uniform prior) | same | same | same |
| r1 (char) | + phoneme CTC auxiliary head | 96.5 | 48.6 | 55.8 | 71.2 / 71.7 | 70.3 / 64.9 | 70.4 / **64.5** [58.5, 69.3] | 63.4 |
| **r1 (phone)** | decode r1 through the pronunciation lexicon | 96.5 [93.3, 98.9] | **54.9** [49.4, 60.2] | **52.0** [48.1, 56.2] | 72.2 / 71.9 | 71.3 / 70.1 | 71.1 / **68.2** [61.4, 73.7] | 71.6 |
| r2 (phone) | r1 + DTW voiced-EMG distillation | 95.3 | 55.0 | 51.7 | 72.2 / 66.3 | 72.1 / 68.2 | 70.8 / **62.0** [54.2, 68.4] | 69.4 |
| r3 (phone) | r1 + stronger augmentation (aug v2); stopped at epoch 26 of 80, stuck on the CTC plateau (dev CER 68.7%) | 7.6 | 8.3 | 92.5 | 13.2 / 13.2 | 13.1 / 12.8 | 13.0 / **12.1** | 11.2 |

Page lexicon sizes (test): page10 mean 19.4 words (10–50), page20 24.9 (20–53), page30 32.2 (30–56).
Secondary page30-hard+lm (trigram/bigram prior over the page words): base 68.3, r1 phone 70.2, r2 phone 69.4.
Isolated-word proxy, r1 phone: iso10/20/30 easy 98.3 / 97.3 / 96.7, hard 78.6 / 73.4 / 71.6; dates hard 95.9 at every N.

Paired tests (WER, B − A, 95% CI, one-sided p):
- r0 trigram vs bigram (base): open −1.5 [−3.2, +0.1] p = 0.04; closed500 −0.5 [−2.0, +1.1] p = 0.30; closed100 +10.8 (n = 10). Kept bigram as the default.
- r1 vs base (both char): open −2.4 p = 0.04; page30-hard −4.6 [−10.6, +2.9] p = 0.10; page10-hard −6.5 p = 0.002.
- r1 phone vs r1 char (same checkpoint): closed500 −6.3 [−9.1, −3.4] p < 0.001; open −3.8 p = 0.006; page30-hard −3.7 [−7.0, −0.3] p = 0.02.
- **r1 phone vs base char: closed500 −9.6 [−15.0, −4.3] p = 0.001; open −6.2 [−9.2, −3.0] p < 0.001; page30-hard −8.3 [−14.5, −0.1] p = 0.02.**
- r2 vs r1 (both phone): page30-hard +6.2 [+2.8, +10.4] (worse), open −0.3 (n.s.). Voiced distillation not kept.

Best: **r1 with phone-lexicon decoding**. Headline page30-hard = **68.2% [61.4, 73.7]**, far from the 90% target.
Everything here is one speaker on lab hardware (see "What these numbers are and are not").

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

## Afono DOM Model: DOM-0 fixture and DOM-1 real pages
`python -m semg.dom_eval` runs both in headless Chrome (CDP, 1280x800, network blocked) and logs them to the registry.

**Model.** DOM Model v1.1 (`extension/dom-model.js`), rules only: candidates from native controls, ARIA widget roles,
click handlers and pointer-cursor elements (open shadow roots included; closed ones via `chrome.dom` in the extension;
iframes out of scope); filters for rendering, disabled, aria-hidden/inert and occlusion; W3C accname 1.2 names
(aria-labelledby, aria-label, label/for and wrapping labels with embedded controls, alt, svg title, title, placeholder,
name from content); a 1-2 word spoken label (stop words, counts and punctuation dropped, visible text preferred over a
differing aria-label, bare numbers become "link two"); duplicates numbered in reading order ("reply one", "reply two");
every target also has a badge number. The page lexicon = label words + the Tier 1 command words that apply
(scroll words only if the page scrolls, editing words only with a field, answers only with a dialog) + the number words
needed for the badges. It goes to Nullpoint as the flat `vocabulary` list (unchanged format) plus a split `lexicon`.

**DOM-0 fixture v3** (`extension/test/fixture.html`): 48 true targets and 12 decoys. Adds to v1: aria-labelledby (incl. a
hidden referenced node and a missing id), label/for, a wrapping label with an embedded text field, aria-labelledby that
includes an input value, icon-only buttons (title, svg title, none), duplicates (3x Reply, 3x Share, 2 unnamed icon
buttons), nesting (card with onclick holding a button, pointer card wrapping a link, button inside a link, span inside a
button), a display:contents link, a custom checkbox with a transparent input, open shadow DOM; decoys: sr-only skip link,
focusable scroll region, named anchor, aria-disabled, opacity 0, inert. Closed shadow DOM and iframes are present but
not scored (out of scope in the harness). v1.1: precision 100%, recall 100%, label accuracy 100%, accname checks 18/18,
labels unique, 8/8 commands. The old v0 scored precision 97.7%, recall 91.5% and labels unique = no on fixture v2 (same page without the display:contents case) (its
label accuracy, 58%, is not a fair comparison: the expected labels follow the v1 scheme). We wrote this page ourselves,
so it is a regression test, not evidence of real-world performance.

**DOM-1 real pages v1** (`extension/test/pages/`): 18 public pages saved 2026-09-29 by `python -m semg.dom_eval snapshot`
(scripts removed, CSS inlined and pruned, images blanked to the same size, web fonts dropped; sources and notes in
`SOURCES.json`): Wikipedia article and main page, GitHub repo and login, Hacker News and its login, BBC News, Bing and
DuckDuckGo results, Python docs, MDN, npm, USA.gov, Craigslist, weather.gov, an Allbirds product page, a books demo
shop, an arXiv abstract. Stack Overflow (bot check) and Reddit (login wall that does not render without scripts) were
excluded. BBC and Allbirds open a full-screen cross-origin consent/marketing iframe on first visit; as captured the
model finds 0 targets there (it cannot reach iframes), so the overlay was removed at capture. Ground truth: every
visible, enabled control in the first viewport of 7 pages (BBC, Bing, GitHub login, npm, Allbirds, USA.gov,
weather.gov; 173 targets), labelled by the Claude agent from screenshots, a broad DOM candidate list and point probes,
not by an independent human (`<page>.gt.json`).

| DOM-1, v1.1 (18 pages) | first viewport | whole page |
|---|---|---|
| actionable targets per page, median (range) | 33 (4-219) | 70 (4-546) |
| targets with a speakable accessible name | 95.1% [89.9, 98.2] | 92.3% [88.7, 96.4] |
| labels unique after numbering | 100% | 100% |
| label words per page ("words on screen"), median (range) | 40 (4-248) | 82.5 (4-383) |
| pages with <= 30 label words | 6 / 18 | 3 / 18 |
| same with one-word labels, median; pages <= 30 | 28.5; 9 / 18 | 56.5; 4 / 18 |
| full lexicon incl. commands and badge numbers, median | 84.5 | 126.5 |

Ground truth (7 pages, 173 targets, first viewport): v1.1 precision 99.4% (page-bootstrap CI [98.2, 100]), recall 100%;
v1 (frozen before labelling) and v0 both 99.4% / 99.4%. The one v1 miss was a display:contents link (BBC); the v1.1 fix
was made after seeing the test labels, so the v1.1 recall is tuned on test. The one false positive is a hover-only
permalink icon on npm. CIs are percentile bootstraps over pages (2,000 resamples; element-level CIs in the registry).
These pages are well-built and semantic, so detection is close to saturated here; they say nothing about div-soup sites,
which is what the learned detector is for (plan in `extension/test/webui/webui_loader.py`). Labels vs. v0 on the same
18 pages: unique 92.5% -> 100%, named 93.4% -> 95.1%, median label words 46.5 -> 40.

Reading: the "10 to 30 words" target holds for simple pages (logins, search results, product and gov pages) but not for
link-heavy pages (Hacker News 123, Craigslist 248, Wikipedia 79-103 in the first viewport). Next levers: restrict the
lexicon to the gaze region, one-word labels, and badge numbers for long lists.

## Afono Track Model v1: session protocol
Pipeline (`app/static/track.js`): MediaPipe Face Landmarker iris landmarks 468-477 -> eye-in-head features (iris
centre relative to the eye corners, both eyes, aspect-corrected) + head pose from the facial transformation matrix ->
ridge-regularised 2nd-order polynomial -> One Euro filter -> dispersion fixation detector -> Bayesian snap to the
page element with hysteresis.
- **Calibration**: 9 dots in a 3 × 3 grid at 10/50/90% of the screen, random order, 0.6 s settling then 1 s of frames
  per dot; frames more than 3.5 robust SDs from the dot's median are dropped; a dot with < 10 clean frames is repeated
  once; one dot is dropped if its leave-one-point-out (LOPO) error is > 2.5 × the median. The mapping (poly2 vs the v0
  linear map) and the ridge lambda are chosen by LOPO on calibration data only.
- **Accuracy** = distance between each validation target and the median estimate over 1 s (after 0.6 s settling),
  averaged over 5 targets not used in calibration. **Precision** = RMS distance between consecutive estimates. Both in
  pixels and degrees from the entered screen width and viewing distance. Both mappings are scored on the same frames.
- **Snap sigma** = sqrt(mean |error|² / 2) over all smoothed validation estimates (isotropic Gaussian fit); before the
  accuracy test, the LOPO error is used.
- **Snap rule**: p(element | gaze) ∝ prior × (Gaussian(sigma) convolved with a uniform box over the element), plus a
  "none" hypothesis (5% prior, uniform over the screen). The highlight switches only when a challenger has ≥ 1.5 × the
  current probability for ≥ 120 ms (≥ 300 ms to drop to none). Dwell = same highlight for 800 ms inside a fixation.
- **Target test**: 12 cells in a 4 × 3 grid, random order, selection by Space when the highlight is on the goal.
  Reports snap hit rate and, on the same trials, the hit rate of a plain gaze cursor (smoothed point inside the goal)
  and of nearest-centre snapping, plus time per selection.
- **Output for the voice decoder**: `GET /api/track/candidates` (or `AfonoTrack.candidates(k)` in the page):
  `{gaze, sigma_px, highlight, p_none, fixating, candidates: [{id, label, p}]}`, top 10 elements. Page rects come in
  through `AfonoTrack.setTargets([{id, label, rect: {left, top, width, height}}])` in screen CSS pixels.

No human session has been logged yet.

## Afono Track Model: snapping simulation (SIMULATION, not a measurement)
`python -m semg.gaze_eval sim --variants`. Layouts are real: the DOM-0 fixture and 18 DOM-1 page snapshots
(`extension/test/pages/`, first 1920 × 1080 viewport, elements found by the DOM Model), plus 5 synthetic layouts
(32 px toolbar, 44 px grid, search results and mail list with repeated labels, the 12-cell target test). The gaze
error is simulated: isotropic Gaussian scaled so the mean error is 1-5° on a 1920 px / 34 cm screen at 60 cm
(1° = 59 px); the user looks inside the central 50% of the target; sigma is estimated from a simulated 5-point
validation. 200 trials per element. CIs: bootstrap over layouts (pages), 2,000 resamples, seed 1234.
Voice: an assumed 90%-accurate word decoder, errors uniform over the page vocabulary (not a measured confusion matrix).

Real pages (19 layouts, equal weight), hit rate [95% CI]:

| Mean gaze error | Pixel cursor | Snap (Bayes) | Snap top-3 contains target | Gaze × voice (all elements) | Voice alone |
|---|---|---|---|---|---|
| 1° | 15.7% [12.7, 18.9] | 52.7% [45.3, 59.8] | 84.5% [76.7, 90.6] | 94.4% [93.8, 95.1] | 90.1% |
| 2° | 5.9% [4.5, 7.4] | 32.5% [26.5, 38.5] | 64.3% [54.5, 73.2] | 92.1% [91.6, 92.6] | 90.0% |
| 3° | 3.1% [2.3, 4.1] | 23.5% [18.5, 28.5] | 51.9% [42.0, 61.0] | 91.1% [90.8, 91.5] | 89.9% |
| 4° | 2.0% [1.4, 2.6] | 18.6% [14.2, 22.6] | 43.3% [34.1, 52.2] | 90.7% [90.5, 91.0] | 90.0% |
| 5° | 1.4% [1.0, 1.8] | 15.4% [11.6, 19.0] | 37.5% [28.6, 46.1] | 90.5% [90.4, 90.7] | 90.1% |

Snap minus pixel (same trials): +37.1 points [30.9, 42.7] at 1°, +26.6 [21.4, 31.9] at 2°, +14.0 [10.5, 17.6] at 5°.

Reading:
- Snapping beats pixel steering at every error level (3× the hit rate at 1°, 5× at 2°, 11× at 5°). On its own it is
  not enough: at a realistic 2-3° webcam error the snapped element is right only a quarter to a third of the time,
  because real pages are dense (Hacker News: 215 targets, median 56 × 11 px).
- The Bayesian rule and nearest-centre snapping pick the same element almost always (within 1.3 points in every
  variant); the value of the posterior is calibrated probabilities for fusion, not a better argmax.
- Hard "top-3, then voice" is worse than voice alone on real pages (60.7% vs 90.0% at 2°), because the top 3 often
  misses the target. Soft fusion (gaze posterior × voice likelihood over all elements) beats voice alone (92.1% vs
  90.0% at 2°), and by more where labels repeat (synthetic search results 59.9% vs 39.6%, mail list 44.4% vs 32.7%).
  So the decoder should fuse softly over the top 10 candidates, not truncate to 3.
- Sensitivity: 50% of the error as a per-session constant bias, or fixations uniform over the whole element, move
  the pooled numbers by at most 4 points.

## Afono Track Model: MPIIFaceGaze
Dataset: Zhang, Sugano, Fritz & Bulling, "It's written all over your face" (CVPRW 2017). 15 participants, 37,667
annotated laptop-webcam images (1,498-2,929 per person) from everyday use over many days, with 3D gaze targets and
camera calibration. Licence CC BY-NC-SA 4.0 (non-commercial). Original images: 942 MB, no registration,
`https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip`; the normalised 448 × 448 version is 22.3 GB. Protocol:
leave-one-subject-out (15 folds), mean angular error between predicted and true 3D gaze directions (from the face
centre). The paper's full-face CNN reports 4.8°.

Our landmark baselines (`python -m semg.gaze_eval mpii-extract` then `mpii`; MediaPipe Face Landmarker on CPU, face
found with open eyes in 37,555 images; ridge regression, lambda from an inner split of training subjects):

| Model | Mean angular error (per-subject range) |
|---|---|
| Predict the mean gaze direction (no model) | 12.0° (9.7-16.6) |
| Nullpoint features (poly2 eye + head pose), person-independent, LOSO | 6.3° (5.2-8.2) |
| Eye landmarks in the head frame + head pose, person-independent, LOSO | 6.2° (5.1-9.6) |
| Nullpoint poly2, personal calibration on 9 images nearest a 3 × 3 grid | 9.1° (5.9-13.1) |
| Nullpoint poly2, personal calibration on 9 × 5 images | 5.0° (3.4-6.3) |
| Population model (LOSO) + per-person offset from 9 images | 5.7° (3.8-11.6) |
| Population model (LOSO) + per-person offset from 9 × 5 images | 4.9° (3.7-6.7) |

The personal rows test on the same person's other images, often from other days and head poses, which is harder than
a live session right after calibration. Found while building this: the v0 features averaged the horizontal iris
position of the two eyes measured from each eye's inner corner, which cancels for sideways gaze; measuring both eyes
image-left to image-right took the poly2 LOSO error from 9.5° to 6.3°.

**Input cap (2026-09-30).** The mapping standardises each input by its spread during calibration. Head pose barely
moves while calibrating, so a small head movement afterwards became a huge z-score and threw the estimate off-screen
(seen in the first human session: 2.9° right after calibrating, 55° in another run, and no snapping minutes later).
Standardised inputs are now capped at ±3. Same protocol as above, with and without the cap:

| Personal poly2 calibration | Mean error | 95th percentile | Worst image |
|---|---|---|---|
| 9 images, no cap → cap 3 | 9.10° → 8.52° | 18.7° → 17.0° | 142.7° → 57.9° |
| 9 × 5 images, no cap → cap 3 | 4.95° → 4.91° | 10.6° → 10.4° | 139.9° → 60.5° |

**First human session (founder, 2026-09-30, desktop app, 1536 × 864 CSS px, 34.5 cm wide screen at 60 cm):**
accuracy test 2.94° (137 px), precision 0.69°, 5 of 5 points, eyes tracked 99% of frames, right after calibrating.
One person, one session: not a benchmark. The target test in the same session got no snaps because of the
extrapolation bug above.

**Population head in Nullpoint (2026-10-12).** The table above works in gaze angles. Nullpoint maps to screen pixels,
so the shipped pipeline was scored the same way (`python -m semg.gaze_eval mpii-screen`): per subject, 9 calibration
images nearest a 3 × 3 grid, each mapping fitted to the subject's screen pixels with lambda by leave-one-point-out,
inputs capped at ±3 SD as in track.js, tested on all the subject's other images; pixel errors converted to mm with the
subject's screen size and to degrees with the face-to-target distance. "pop" = the population head (eye landmarks in
the head frame + head pose → pitch, yaw; ridge, trained leave-one-subject-out) followed by a per-person linear map of
(pitch, yaw, tx/z, ty/z) to the screen. "Selected" = whichever has the lower leave-one-point-out error on the 9
calibration points, which is what `calibrate()` does.

| Screen-space calibration | poly2 | pop | selected (pop chosen) |
|---|---|---|---|
| 9 points × 1 image | 8.21° | 7.84° | **7.71°** (12 / 15 subjects) |
| 9 points × 5 images | 5.22° | 5.22° | **5.17°** (4 / 15) |

The pop design (pitch, yaw, tx/z, ty/z) was fixed before this run. Afterwards, three other designs were tried on the
same data and not adopted (choosing among them here would be tuning on test): (pitch, yaw) only 7.44° / 6.14° (pop
column, 9 × 1 / 9 × 5), with quadratic terms 8.23° / 6.62°, tangent projection + t/z 7.89° / 5.26°. Reading: the
population head helps when calibration data is thin (one image per point) and adds nothing once there are five; the
gain is smaller than the angle-space rows suggest. Shipped as `app/static/track_pop.json` (5 KB, trained on all 15
subjects; `python -m semg.gaze_eval export-pop`); tests/test_track.py checks the JS and Python heads agree to 1e-9.
The weights derive from MPIIFaceGaze (CC BY-NC-SA 4.0), so they are for non-commercial use.

Model plan: (1) done, personal poly2 calibration in the browser; (2) done (above), a population regression head on MediaPipe
landmarks trained on MPIIFaceGaze, mapped per person from the 9-point calibration;
(3) later, a small CNN on normalised eye patches (MobileNet-sized, ONNX in the browser), trained leave-one-subject-out
on MPIIFaceGaze with the paper's 4.8° as the bar, then fine-tuned per person.

## Whisper mode: quiet speech from an ordinary microphone
`semg/voice.py`. Whisper base.en (Radford et al., 2022) scores every phrase the page allows (labels, `click` plus a
label or badge number, Tier 1 commands) by teacher-forced log-likelihood. The encoder runs once per clip; the free
transcription acts as a garbage model. A phrase is **accepted** when its posterior (uniform prior, optionally
re-weighted by Track Model gaze with a uniform floor) is at least 0.7 and the free transcription either agrees with
it (character similarity >= 0.75, so homophones like "reply to" / "reply two" agree) or scores at most 4 nats higher.
Thresholds were set on synthetic clips, not on test data from people.
**Change (2026-10-12):** the posterior is summed over phrases that send the same command before the 0.7 test
("click twenty five" and "click two five", or "reply" and "click reply", are one action). Badge numbers can now be
said digit by digit. The synthetic check above was run before this change; self-tests logged from today use it.

**Pipeline check (SYNTHETIC, not a measurement):** Windows SAPI text-to-speech in a normal voice, 36 prompts on
three page-like phrase sets (median 62 phrases): 36/36 top choice correct, none rejected. Off-page speech
(4 sentences), silence and two noise levels: all rejected. Median 1.1 s per clip on the laptop CPU (as in the
desktop build), 0.3-0.6 s on the RTX 4060. This shows the pipeline works; it says nothing about whispers or real
voices.

Speed: the prompt tokens and the cross-attention keys and values over the audio are computed once per clip and
shared by every phrase (scores identical to scoring each phrase separately, checked to 1e-5). Scoring against only
the encoder frames covering the clip was tried and rejected: CPU accuracy on the same check fell to 28/36.

**Self-test (the measurement):** Nullpoint's Whisper tab draws 20 prompts from the connected page's phrases.
It logs, per session, top-choice accuracy, accuracy when accepted, rejection rate, speaking style (whisper,
quiet, normal) and microphone, as `voice-1` in the run registry.

## Gaze × sEMG fusion in Nullpoint (2026-10-12, mechanism only)
The Track simulation above favours soft fusion over all elements. Nullpoint's sEMG decoder now applies it to the page
vocabulary (`semg.page.gaze_word_weights`): each target gets p' = 0.5 × its gaze probability + 0.5 / (number of
targets); a label word's prior is multiplied by n × the mean p' of the targets whose label contains it (1 when gaze is
uniform, never 0); command and number words are unchanged. Gaze older than 1.5 s is ignored, and a request can turn it
off (`"gaze": false`). It is the rule Whisper mode already uses for phrases. Not yet measured: needs Tier 1 recordings
made while looking at a page.

## Planned protocols for the other two models
- **Afono Track Model** (gaze): first human sessions with the v1 protocol above (accuracy, sigma, snap vs pixel hit rate on the same trials); then a CNN on MPIIFaceGaze, leave-one-subject-out.
- **Afono DOM Model** (page elements): DOM-1 above; next, precision and recall of clickable elements on the WebUI test split and on a hand-labelled DOM-2 set of div-soup pages (the case the learned detector targets).
- **Combined system**: success rate and time per task on scripted web tasks, selections per minute and ITR, against keyboard, mouse and eye-gaze baselines. Participants with and without motor impairments, with ethics review before any human study.

## Rules
1. Change one thing per run and say what in `--notes`.
2. Never tune on test. A protocol change gets a new version number here and in `semg/evaluate.py`.
3. Report every run, including the ones that got worse.
