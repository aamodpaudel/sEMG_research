import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming, springTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { AppScene, Audience, FusionScene, GazeScene, Hook, Intro, Outro, PageScene, SemgScene, ThreeModels, WhisperScene } from "./scenes";

export const SCENES: { c: React.FC; d: number }[] = [
  { c: Intro, d: 126 },
  { c: Hook, d: 132 },
  { c: Audience, d: 150 },
  { c: ThreeModels, d: 216 },
  { c: PageScene, d: 210 },
  { c: GazeScene, d: 200 },
  { c: SemgScene, d: 210 },
  { c: WhisperScene, d: 240 },
  { c: FusionScene, d: 180 },
  { c: AppScene, d: 165 },
  { c: Outro, d: 170 },
];
export const T = 16;
export const TOTAL = SCENES.reduce((a, s) => a + s.d, 0) - T * (SCENES.length - 1);

export const AfonoFilm: React.FC = () => (
  <AbsoluteFill style={{ background: "#F7F9F9" }}>
    <TransitionSeries>
      {SCENES.map((s, i) => (
        <React.Fragment key={i}>
          <TransitionSeries.Sequence durationInFrames={s.d}><s.c /></TransitionSeries.Sequence>
          {i < SCENES.length - 1 ? (
            <TransitionSeries.Transition
              presentation={i % 3 === 2 ? slide({ direction: "from-right" }) : fade()}
              timing={i % 3 === 2 ? springTiming({ config: { damping: 200 }, durationInFrames: T }) : linearTiming({ durationInFrames: T })}
            />
          ) : null}
        </React.Fragment>
      ))}
    </TransitionSeries>
  </AbsoluteFill>
);
