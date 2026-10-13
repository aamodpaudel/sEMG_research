import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadManrope } from "@remotion/google-fonts/Manrope";
import { loadFont as loadMono } from "@remotion/google-fonts/IBMPlexMono";

const inter = loadInter("normal", { weights: ["400", "500", "600", "700"], subsets: ["latin"] });
const manrope = loadManrope("normal", { weights: ["800"], subsets: ["latin"] });
const mono = loadMono("normal", { weights: ["400", "500"], subsets: ["latin"] });

export const FONT = {
  text: inter.fontFamily,
  mark: manrope.fontFamily,
  mono: mono.fontFamily,
};

// BRAND.md tokens (light look).
export const C = {
  teal: "#0F5E63",
  teal2: "#137A80",
  tealSoft: "#5DB5AE",
  signal: "#D9552B",
  ink: "#12262A",
  text: "#1D1D1F",
  text2: "#45484D",
  text3: "#6B6F75",
  paper: "#F7F9F9",
  mist: "#E4EBE8",
  white: "#FFFFFF",
  glass: "rgba(255,255,255,0.62)",
  line: "rgba(22,24,26,0.10)",
};

export const FPS = 30;
export const W = 1920;
export const H = 1080;
