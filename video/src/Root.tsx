import React from "react";
import { Composition } from "remotion";
import { AfonoFilm, TOTAL } from "./Composition";
import { FPS, H, W } from "./theme";

export const RemotionRoot: React.FC = () => (
  <Composition id="AfonoFilm" component={AfonoFilm} durationInFrames={TOTAL} fps={FPS} width={W} height={H} />
);
