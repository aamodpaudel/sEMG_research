# Related work: silent commands, earables, gaze and page-constrained decoding

Literature scan, 2026-10-12. **V** = number read in the paper (full text or abstract). **S** = from a secondary
source or snippet: check before citing. Numbers are as the papers report them, under their own protocols.

## 1. What the user must do, and what can sense it
| User action | Visible? | What can sense it | Best evidence |
|---|---|---|---|
| Mouth words (lips/jaw move, no sound) | Yes | Camera, earbud ultrasound or IMU, electrodes | Many systems, 90–96% on 8–100 commands, per-user models |
| Closed lips, faint tongue/throat articulation | Barely | Electrodes under the chin and beside the larynx | AlterEgo 92% (10 digits), NASA 92% (6 words) |
| Teeth tap / click / slide, jaw closed | No | Earbud mic, speaker or IMU (bone-conducted vibration) | NeuroClick 98.6% macro-F1 cross-user; EarSense >90% on unmodified earphones |
| Pure inner speech (no articulation) | No | Nothing non-invasive; surface EMG falls to chance | Nalborczyk 2020: 84.7% overt vs 47.2% inner (2 classes, chance 50%) |
| Imagined words, EEG | No | Full-scalp EEG, ~30–60% on 2–6 words | Alzahrani 2024 review |
| Attempted speech, implant | No | Intracortical arrays / ECoG | Card 2024: 99.6% on 50 words, day 1 |

**No earbud system recognises speech with zero articulator movement.** The closest invisible earbud input is teeth
gestures. Closed-lip speech needs electrodes under the chin or on the throat, not in the ear.

## 2. Earables (earbuds, earphones, headphones)
| Work | Hardware | User action | Vocabulary | Result | Note |
|---|---|---|---|---|---|
| EarCommand, Jin et al., IMWUT 2022, [doi](https://doi.org/10.1145/3534613) | Custom earbud, inward speaker+mic, near-ultrasound | Mouthing | 32 words, 25 sentences | WER 10.0% / 12.3% (V, abstract) | Ear-canal deformation |
| MuteIt, Srivastava et al., IMWUT 2022, [doi](https://doi.org/10.1145/3550281) | Custom: 2 IMUs on an ear hook, one taped at the jaw joint | Slow mouthing | 100 commands | 94.8% quiet, 91% moving; >90% cross-session (V) | Says AirPods/eSense IMU is in the wrong place |
| Unvoiced, Srivastava et al., SenSys 2024, [doi](https://doi.org/10.1145/3666025.3699374) | Same twin-IMU prototype + Whisper + LLM | Mouthing | Phrases, 4 tasks | >94% task completion (V) | LLM-assisted |
| HPSpeech, Zhang et al., ISWC 2023, [doi](https://doi.org/10.1145/3594738.3611365) | Bose QC45 speakers + added mic (built-in mic not accessible) | Exaggerated mouthing | 8 commands | 90–92% per user; 46.9% cross-user (V) | Low-movement user: 67.7% |
| ReHEarSSE, Dong et al., CHI 2024, [doi](https://doi.org/10.1145/3613904.3642095) | Earbud ultrasound | Silent spelling | 100 unseen words | 89.3% (V, abstract) | |
| EarSSR, Sun et al., IEEE TMC 2024, [doi](https://doi.org/10.1109/TMC.2024.3356719) | Earphone ultrasound | Silent articulation | Letters, words | 82% / 93% (S) | |
| Trumpet of the Swan, Gao et al., IMWUT 2025, [doi](https://doi.org/10.1145/3770648) | **Unmodified AirPods IMU (25 Hz)** | Lip mouthing | (not read) | (not read) | 11 speech-impaired users; only commodity speech work found |
| Whispering Wearables, Srivastava et al., ICMI 2024 (Microsoft) | Head-worn IMU near ear + ExG | Silent speech | 12 commands | 94.2%, 5 samples/user to adapt (S) | |
| EarSense, Prakash et al., MobiCom 2020, [doi](https://doi.org/10.1145/3372224.3419197) | **Unmodified wired earphones**, speaker re-tasked as sensor (Realtek jack retasking) | Teeth tap/slide, jaw closed | 6–7 gestures | >90% (V) | Invisible |
| TeethTap, IUI 2021, [arXiv](https://arxiv.org/abs/2102.12548) | Custom earpiece IMU + contact mic | Teeth gestures | 13 gestures | 90.9% real time (V) | |
| NeuroClick, Hummel et al., arXiv 2026, [arXiv](https://arxiv.org/abs/2609.12910) | OpenEarable 2.0 | Tooth clicks | Click commands | 98.6% median macro-F1, leave-one-subject-out (V) | Cross-user, invisible |
| EarBuddy, Xu et al., CHI 2020, [doi](https://doi.org/10.1145/3313831.3376836) | Commercial earbud mic | Taps on the face (hand) | 8 gestures | 95.3% (V) | Not hands-free |
| EchoSpeech, Zhang et al., CHI 2023, [doi](https://doi.org/10.1145/3544548.3580801) | Glasses with speakers+mics (sonar) | Mouthing | 31 commands, digits | WER 4.5% / 6.1% (V) | Near-commodity glasses |

**Developer access.** AirPods (`CMHeadphoneMotionManager`) give only processed head motion, about 25 Hz in papers;
no in-ear mic or speaker access. Android head trackers give orientation and angular velocity. Research earbuds give
raw access: eSense (IMU up to 50 Hz) and **OpenEarable 2.0** (open source, KIT: inner and outer mics at 48 kHz,
bone-conduction accelerometer, 9-axis IMU, ear-canal pressure), [site](https://open-earable.teco.edu/).

## 3. Electrodes: subvocal, near-ear, neck
| Work | Sensors | User action | Vocabulary | Result |
|---|---|---|---|---|
| AlterEgo, Kapur et al., IUI 2018, [doi](https://doi.org/10.1145/3172944.3172977) | 7 wet channels (chin, larynx, hyoid, cheek), OpenBCI 250 Hz | "Without sound and moving their lips" (not measured) | Digits 0–9 | 92.01%, 10 users, per-user, random split, no re-wear test (V) |
| Jorgensen et al., NASA Ames 2003, [ntrs](https://ntrs.nasa.gov/citations/20030018896) | 2 wet pairs: under chin, beside larynx | Faint, "little if any external movement" | 6 words | 92%, 3 subjects, 6 days (V) |
| Nalborczyk et al., PLOS ONE 2020, [doi](https://doi.org/10.1371/journal.pone.0233282) | Facial EMG | Overt vs inner speech | 2 classes | 84.7% vs 47.2% (chance) (V); data [OSF](https://osf.io/czer4) |
| Tang et al. (Cambridge), arXiv 2025, [arXiv](https://arxiv.org/abs/2504.13921) | **4 dry textile electrodes in headphones, ESP32-S3, 1 kHz** | "Silent speech" (mouthing not stated) | 10 words (open, close, start, stop, yes, no, next, back, okay, cancel) | 96%, 4 subjects, pooled random split (V) |
| AESSI, Xu et al., arXiv 2026, [arXiv](https://arxiv.org/abs/2609.21436) | Two cEEGrids around the ears (18 ch, wet) | Mouthing | 25 sentences | 92.2% cross-day without calibration; 98% online after 21 days (V) |
| Inoue et al., J. Neural Eng. 2026, [doi](https://doi.org/10.1088/1741-2552/ae54d0) | cEEGrid 16 ch | Mouthed / attempted (locked-in user) | 64 words | 56.6% healthy; signal is EMG mixed with EEG (V) |
| SilentWear, Spacone et al., arXiv 2026, [arXiv](https://arxiv.org/abs/2603.02847) | 14-ch dry textile neckband | Voiced vs silent | 8 commands | 84.8% / 77.5% in session; **71.1% / 59.3% after re-wearing** (V); [code](https://github.com/pulp-bio/SilentWear), data on HF |
| Le et al. (NYU), arXiv 2026 | 8 ch face/neck | Aloud / mimed / occluded subvocal | TIMIT | Subvocal above chance, lower than mimed (V, summary) |

Context: implants (Willett, Nature 2023: 9.1% WER on 50 words; Card, NEJM 2024: 99.6% on 50 words; Metzger, Nature
2023: 78 wpm). Non-invasive imagined speech: ~30–60% on 2–6 words (Alzahrani, Sensors 2024).

## 4. Gaze plus a confirming channel, and page-constrained decoding
| Work | What | Result |
|---|---|---|
| Jacob, CHI 1990, [doi](https://doi.org/10.1145/97243.97246) | Names the Midas touch problem | Qualitative |
| Surakka et al., ACM TAP 2004 | Gaze + frown EMG click | (not verified) |
| Mateo et al., CHI EA 2008, "Gaze beats mouse" | Gaze + facial EMG select | Gaze pointing faster than mouse (V, abstract) |
| Rozado et al., TACCESS 2017 (FaceSwitch), [doi](https://doi.org/10.1145/3075301) | Gaze + webcam face gestures to several actions | 1.6 s mouse, 2.5 s gaze+FaceSwitch, 3.9 s gaze only (S) |
| Miniotas et al., ETRA 2006 | Gaze + spoken nudge commands, small targets | Errors 51.1% → 17.4% (V) |
| **Pandey & Arif, PACM HCI (ISS) 2022**, [doi](https://doi.org/10.1145/3567723), [code](https://github.com/theiilab/Eye-Gaze-Pointing) | **Webcam gaze + webcam lip-read silent commands** | ~95% with 6 keywords; menu task errors 51.1% → 1.9% (V). **Closest prior work** |
| Su et al., ETRA 2021 (Gaze+Lip), [doi](https://doi.org/10.1145/3448018.3458011) | Gaze + lip-read commands, smart TV | Better than gaze-only (abstract) |
| Gemicioglu et al., CHI EA 2023 (Gaze & Tongue) | Gaze + tongue gestures from ear IMU/PPG/EEG | Demo |
| Stawicki et al., Brain Sci. 2017, [doi](https://doi.org/10.3390/brainsci7040035) | Gaze + SSVEP hybrid speller | Serves more users (90.6% vs 78.1%), slower (V) |
| Meta, Nature 2025, [doi](https://doi.org/10.1038/s41586-025-09255-w) | Wrist sEMG, generic across users | 0.88 gesture detections/s, 20.9 wpm (S) |
| Papoutsaki et al., IJCAI 2016 (WebGazer) | Webcam gaze | ~4° in person, 175–233 px online (V) |
| Li et al., GI 2021 (BayesGaze) | Bayesian target selection from gaze | 82.1% → 88.3% vs dwell (V) |
| Menges et al., W4A 2017 / ToCHI 2019 (GazeTheWeb) | Gaze browser using the DOM | 252 s vs 424 s for OptiKey+Chrome (S) |
| Christian et al., ASSETS 2000 | Voice vs mouse browsing, numbered links | Users preferred text links to numbers (V) |
| Heck et al. 2013; Slaney et al., ICASSP 2014 | Pointing or gaze constrains speech recognition to page links | 16% and 25% relative WER reduction (S / V) |
| P300 web browsers (Mugler 2010, Halder 2015, Martínez-Cagigal 2016) | EEG browsing | 58–96%, slow (S) |

## 5. Where Afono is new (not found in the literature)
1. **Gaze + an EMG or earable silent word vocabulary.** Gaze + silent speech exists only with camera lip reading
   (Pandey & Arif 2022, Gaze+Lip 2021). Gaze + EMG papers use a single "click", not words.
2. **A decoder constrained by the DOM / accessibility tree and the actions the gazed element allows**, applied to
   silent input. Page-constrained grammars exist for voice (1995 onwards; Slaney 2014 with gaze), not for silent input.
3. **Soft Bayesian fusion of webcam-gaze target probabilities with command likelihoods in a real browser**
   (BayesGaze covers gaze only; GazeTheWeb is dwell-based on a dedicated tracker).
4. **A peer-reviewed comparison with "show numbers" voice overlays** (Voice Control, Voice Access): none since 2000.
5. **A webcam-plus-earable hands-free web stack evaluated with motor-impaired users on real sites.**

## 6. What this means for the product
- **Invisible + earbud + works across users today = teeth gestures** (tap, double tap, slide), on OpenEarable 2.0 or
  re-tasked wired earphones, with gaze choosing the element. A handful of gestures covers click, double, right,
  scroll. They are not speech, but they give the "it reads my mind" feel with nothing visible.
- **Invisible words = closed-lip articulation with an electrode under the chin** (AlterEgo / NASA sites), e.g. an
  ear hook with a chin arm on the existing ESP32 rig. Per-user models; plan a short re-calibration on each wear
  (SilentWear lost ~18 points after re-wearing).
- **Not possible without surgery:** pure thought. Do not claim it.
- **Benchmark against:** Pandey & Arif 2022 (gaze + silent lip commands), FaceSwitch 2017, Voice Access / Voice Control.
