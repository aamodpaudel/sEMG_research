# Afono film

A 61-second product film (1920 × 1080, 30 fps) made with [Remotion](https://www.remotion.dev/). It walks through
the three models (DOM, Track, sEMG), Whisper mode, gaze × voice fusion and Afono Nullpoint, in eleven scenes
(`src/Composition.tsx`). Colours and type follow `BRAND.md`; the screen recordings and screenshots are in `public/`.

```console
npm i --loglevel=error
npm run dev                                          # Remotion Studio, live preview
npx remotion render AfonoFilm out/afono-film.mp4     # render (out/ is git-ignored)
npm run lint                                         # eslint + tsc
```

Numbers shown in the film must come from `runs/registry.jsonl`, like every number on the website.
