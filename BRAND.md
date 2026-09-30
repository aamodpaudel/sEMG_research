# Afono brand

The full, visual guide is `site/brand.html`. This file is the short version.

## The mark
The symbol is the Greek letter **φ**: a ring crossed by a vertical bar.
- **The name.** Afono comes from Greek *áphōnos*, "voiceless", which is spelled with φ.
- **The gesture.** A finger over the lips means "shh".
- **The sensor.** Skin electrodes are round pads; the bar is the signal passing through.

Construction on a 64 × 96 grid: ring radius 22, stroke 9; bar from y 8 to y 88, round caps, same stroke, in front of the ring.
Wordmark: lowercase `afono`, Manrope ExtraBold (800), tracking −4.5%. Clear space = the ring's width on every side.
Minimum size: symbol 16 px tall, lockup 24 px tall.

Files: `site/assets/afono-symbol.svg`, `afono-symbol-light.svg` (dark backgrounds), `afono-symbol-mono.svg`, `afono-icon.svg`.

## Colour
| Name | Light | Dark theme | Use |
|---|---|---|---|
| Afono Teal | `#0F5E63` | `#5DB5AE` | Ring, headings, links, data |
| Signal Orange | `#D9552B` | `#F0794A` | The bar, one call to action per view |
| Ink | `#12262A` | — | Text, dark surfaces, app icon |
| Paper | `#F3F5F2` | `#0D1B1E` | Page ground |
| Mist | `#E4EBE8` | `#15282C` | Quiet surfaces |

## Type
- Wordmark only: **Manrope** ExtraBold (800).
- Display and text: the system face. **SF Pro** on Apple devices, **Inter** everywhere else (the closest open equivalent; SF Pro cannot be hosted on the web). Headlines 600–700 with tight tracking.
- Data and labels: **SF Mono** on Apple devices, **IBM Plex Mono** elsewhere.
- Look: white and #F5F5F7 surfaces, near-black #1D1D1F text, teal and orange used sparingly.
All four are free on Google Fonts. Tokens live in `site/assets/afono.css`.

## Motion: the intro (about 4 s)
1. 0.2–1.0 s: the ring draws itself, like a mouth opening.
2. 1.0–1.5 s: the orange bar drops through it with a small overshoot, the finger saying shh.
3. 1.4–1.8 s: the symbol squashes and settles.
4. 1.8–2.6 s: the symbol slides left and `afono` rises in, 70 ms per letter.
5. 3.0–4.1 s: one thin ring ripples out and fades, a signal without sound.
Skip it entirely when the viewer asks for reduced motion. Keyframes: `site/assets/afono.css`.

## Voice
Plain, warm, exact. Say "reads the muscles of speech", never "reads your mind" or "brain to text".
Always give a vocabulary size next to an accuracy number. Say "people with voice loss", not "sufferers".
Make no medical claims.
