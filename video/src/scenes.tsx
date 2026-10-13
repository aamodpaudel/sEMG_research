import React from "react";
import { AbsoluteFill, Img, OffthreadVideo, interpolate, random, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, FONT } from "./theme";
import { Backdrop, Chip, Eyebrow, Glass, Headline, Phi, Stat, Window, WordStagger, Wordmark, ease, fadeUp } from "./components";
import SHAPES from "./shapes.json";

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

/* ============ 1. Intro: the φ animation from BRAND.md ============ */
export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const ringDraw = interpolate(frame, [6, 30], [0, 1], { ...clamp, easing: ease });
  const barDrop = spring({ frame: frame - 30, fps, config: { damping: 9, stiffness: 140 } });
  const squash = interpolate(frame, [42, 47, 54], [0, 1, 0], clamp);
  const slide = interpolate(frame, [54, 78], [0, 1], { ...clamp, easing: ease });
  const reveal = interpolate(frame, [58, 82], [0, 1], clamp);
  const ripple = interpolate(frame, [88, 122], [0, 1], clamp);
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 36, transform: `translateX(${(1 - slide) * 200}px)` }}>
          <div style={{ position: "relative" }}>
            <Phi size={260} ringDraw={ringDraw} barDrop={barDrop} squash={squash} />
            <div style={{ position: "absolute", left: "50%", top: "50%", width: 150, height: 150, marginLeft: -75, marginTop: -75, borderRadius: "50%",
              border: `3px solid ${C.teal}`, opacity: (1 - ripple) * (ripple > 0 ? 0.6 : 0), transform: `scale(${1 + ripple * 2.4})` }} />
          </div>
          <div style={{ width: slide * 720, overflow: "hidden", paddingRight: 10 }}><Wordmark size={230} reveal={reveal} /></div>
        </div>
        <div style={{ position: "absolute", bottom: 250, ...fadeUp(frame, 86) }}>
          <Eyebrow>Hands-free computing</Eyebrow>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 2. Hook ============ */
export const Hook: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", textAlign: "center", padding: 120 }}>
        <WordStagger text="Use a computer" start={4} size={150} per={5} />
        <WordStagger text="without lifting a finger." start={20} size={150} per={5} accent={["finger"]} />
        <div style={{ marginTop: 56, fontFamily: FONT.text, fontSize: 42, color: C.text2, ...fadeUp(frame, 62) }}>
          Your eyes, your lips and the page, working together.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 3. Who it is for ============ */
export const Audience: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <Backdrop tint="#F0B49A" />
      <AbsoluteFill style={{ justifyContent: "center", padding: "0 180px" }}>
        <div style={fadeUp(frame, 0)}><Eyebrow>Built first for</Eyebrow></div>
        <Headline size={104} style={{ marginTop: 26, maxWidth: 1400, ...fadeUp(frame, 6) }}>People who can't use their hands or voice.</Headline>
        <div style={{ display: "flex", gap: 22, marginTop: 60, flexWrap: "wrap" }}>
          <Chip delay={30}>Voice loss</Chip>
          <Chip delay={38} color={C.signal}>Spinal cord injury</Chip>
          <Chip delay={46} color={C.teal2}>ALS</Chip>
        </div>
        <div style={{ marginTop: 56, fontFamily: FONT.text, fontSize: 40, color: C.text2, ...fadeUp(frame, 78) }}>
          …then everyone with tired hands.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 4. Three models: φ morphs into an eye, lips and a window ============ */
const nums = (d: string) => (d.match(/-?\d+(\.\d+)?/g) || []).map(Number);
const lerpPath = (a: string, b: string, t: number) => {
  const na = nums(a), nb = nums(b);
  let i = 0;
  return a.replace(/-?\d+(\.\d+)?/g, () => { const v = na[i] + (nb[i] - na[i]) * t; i++; return v.toFixed(2); });
};

export const ThreeModels: React.FC = () => {
  const frame = useCurrentFrame();
  const S = SHAPES as Record<string, string>;
  const steps = [
    { key: "eye", name: "Track Model", what: "Where you look", color: C.teal },
    { key: "lips", name: "sEMG Model", what: "What you mouth", color: C.signal },
    { key: "window", name: "DOM Model", what: "What you can click", color: C.teal2 },
  ];
  const seg = 56, start = 24;
  const idx = Math.max(-1, Math.min(2, Math.floor((frame - start) / seg)));
  const local = frame - start - idx * seg;
  const t = interpolate(local, [0, 20], [0, 1], { ...clamp, easing: ease });
  const from = idx <= 0 ? S.ring : S[steps[idx - 1].key];
  const d = idx < 0 ? S.ring : lerpPath(from, S[steps[idx].key], t);
  const barOpacity = idx < 0 ? 1 : 1 - interpolate(frame, [start, start + 14], [0, 1], clamp);
  const cur = idx < 0 ? C.teal : steps[idx].color;
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ padding: "0 160px", flexDirection: "row", alignItems: "center", gap: 120 }}>
        <div style={{ width: 620, display: "flex", justifyContent: "center" }}>
          <svg width={520} height={520} viewBox="0 0 96 96">
            <path d={d} fill="none" stroke={cur} strokeWidth={4.2} strokeLinejoin="round" />
            {idx >= 0 && steps[idx].key === "eye" ? <circle cx={48} cy={48} r={9 * t} fill={C.teal} /> : null}
            {idx >= 0 && steps[idx].key === "window" ? <line x1={12} y1={34} x2={84} y2={34} stroke={cur} strokeWidth={3} opacity={t} /> : null}
            <line x1={48} y1={14} x2={48} y2={82} stroke={C.signal} strokeWidth={4.2} strokeLinecap="round" opacity={barOpacity} />
          </svg>
        </div>
        <div style={{ flex: 1 }}>
          <div style={fadeUp(frame, 0)}><Eyebrow>One interface · three small models</Eyebrow></div>
          <div style={{ marginTop: 36, display: "flex", flexDirection: "column", gap: 30 }}>
            {steps.map((s, i) => {
              const on = i === idx;
              const shown = frame >= start + i * seg;
              return (
                <div key={s.key} style={{ display: "flex", alignItems: "baseline", gap: 28, opacity: shown ? (on ? 1 : 0.38) : 0,
                  transform: `translateX(${shown ? 0 : 30}px)` }}>
                  <div style={{ width: 16, height: 16, borderRadius: 8, background: s.color, alignSelf: "center" }} />
                  <div>
                    <div style={{ fontFamily: FONT.text, fontWeight: 700, fontSize: 76, letterSpacing: "-0.035em", color: C.text }}>{s.what}</div>
                    <div style={{ fontFamily: FONT.mono, fontSize: 24, color: s.color, letterSpacing: "0.08em" }}>AFONO {s.name.toUpperCase()}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 5. DOM Model, with real footage ============ */
export const PageScene: React.FC = () => {
  const frame = useCurrentFrame();
  const win = interpolate(frame, [0, 26], [0, 1], { ...clamp, easing: ease });
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ padding: "0 110px", flexDirection: "row", alignItems: "center", gap: 70 }}>
        <div style={{ width: 640 }}>
          <div style={fadeUp(frame, 4)}><Eyebrow color={C.teal2}>Afono DOM Model</Eyebrow></div>
          <Headline size={78} style={{ marginTop: 22, ...fadeUp(frame, 10) }}>Every button gets a name you can say.</Headline>
          <div style={{ marginTop: 28, fontFamily: FONT.text, fontSize: 32, lineHeight: 1.4, color: C.text2, ...fadeUp(frame, 22) }}>
            Read from the page's accessibility tree. Duplicates are numbered: “reply 1”, “reply 2”.
          </div>
          <div style={{ marginTop: 44 }}>
            <Stat delay={40} value="99.4%" label="of controls found, 99.4% precision" note="7 real websites, 173 controls, first screen" />
          </div>
        </div>
        <div style={{ opacity: win, transform: `translateY(${(1 - win) * 60}px) scale(${0.94 + 0.06 * win})` }}>
          <Window width={1030} title="harbor-daily · Afono extension">
            <OffthreadVideo src={staticFile("dom-model.mp4")} muted style={{ width: 1030, display: "block" }} startFrom={30} />
          </Window>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 6. Track Model: gaze snaps to elements ============ */
const GRID = { cols: 4, rows: 3, x: 60, y: 110, w: 190, h: 92, gx: 34, gy: 46 };
const cell = (i: number) => {
  const c = i % GRID.cols, r = Math.floor(i / GRID.cols);
  return { x: GRID.x + c * (GRID.w + GRID.gx), y: GRID.y + r * (GRID.h + GRID.gy) };
};
const LABELS = ["Home", "Search", "Inbox", "Sign in", "Reply 1", "Share", "Reply 2", "Save", "Next", "Back", "Menu", "Help"];

export const GazeScene: React.FC = () => {
  const frame = useCurrentFrame();
  const path = [0, 5, 6, 3, 9, 6];                       // where the viewer looks
  const seg = 30;
  const k = Math.min(path.length - 1, Math.floor(frame / seg));
  const tt = interpolate(frame - k * seg, [0, 12], [0, 1], { ...clamp, easing: ease });
  const a = cell(path[Math.max(0, k - 1)]), b = cell(path[k]);
  const gx = a.x + (b.x - a.x) * tt + GRID.w / 2 + 38 * (random(`x${Math.floor(frame / 2)}`) - 0.5) + 30 * Math.sin(frame / 7);
  const gy = a.y + (b.y - a.y) * tt + GRID.h / 2 + 34 * (random(`y${Math.floor(frame / 2)}`) - 0.5) + 22 * Math.cos(frame / 9);
  let best = 0, bd = 1e9;
  for (let i = 0; i < 12; i++) {
    const p = cell(i);
    const d = Math.hypot(gx - (p.x + GRID.w / 2), gy - (p.y + GRID.h / 2));
    if (d < bd) { bd = d; best = i; }
  }
  const win = interpolate(frame, [0, 22], [0, 1], { ...clamp, easing: ease });
  const bp = cell(best);
  return (
    <AbsoluteFill>
      <Backdrop tint="#9FD4CF" />
      <AbsoluteFill style={{ padding: "0 110px", flexDirection: "row", alignItems: "center", gap: 80 }}>
        <div style={{ opacity: win, transform: `translateY(${(1 - win) * 50}px)` }}>
          <Glass style={{ width: 1000, height: 560, position: "relative", overflow: "hidden" }}>
            <div style={{ position: "absolute", left: 60, top: 42, fontFamily: FONT.mono, fontSize: 20, color: C.text3, letterSpacing: "0.12em" }}>WEBCAM GAZE · SNAPPED</div>
            {LABELS.map((l, i) => {
              const p = cell(i);
              return (
                <div key={l} style={{ position: "absolute", left: p.x, top: p.y, width: GRID.w, height: GRID.h, borderRadius: 18, background: C.white,
                  boxShadow: "0 0 0 1px rgba(22,24,26,.08)", display: "flex", alignItems: "center", justifyContent: "center",
                  fontFamily: FONT.text, fontWeight: 600, fontSize: 30, color: C.text }}>{l}</div>
              );
            })}
            <div style={{ position: "absolute", left: bp.x - 7, top: bp.y - 7, width: GRID.w + 14, height: GRID.h + 14, borderRadius: 24,
              border: `5px solid ${C.teal}`, boxShadow: "0 0 0 10px rgba(15,94,99,.14)" }} />
            <div style={{ position: "absolute", left: gx - 60, top: gy - 60, width: 120, height: 120, borderRadius: 60, background: "rgba(217,85,43,0.10)", border: "2px dashed rgba(217,85,43,.55)" }} />
            <div style={{ position: "absolute", left: gx - 9, top: gy - 9, width: 18, height: 18, borderRadius: 9, background: C.signal }} />
          </Glass>
        </div>
        <div style={{ flex: 1 }}>
          <div style={fadeUp(frame, 4)}><Eyebrow>Afono Track Model</Eyebrow></div>
          <Headline size={74} style={{ marginTop: 22, ...fadeUp(frame, 10) }}>Your eyes pick the element.</Headline>
          <div style={{ marginTop: 26, fontFamily: FONT.text, fontSize: 31, lineHeight: 1.4, color: C.text2, ...fadeUp(frame, 22) }}>
            Any laptop webcam. Webcam gaze wobbles by about a button's width, so Nullpoint snaps to buttons instead of steering a cursor.
          </div>
          <div style={{ marginTop: 40 }}>
            <Stat delay={42} value="4.9°" label="gaze error after a quick calibration" note="MPIIFaceGaze, 15 people, tested on unseen days" />
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 7. sEMG Model (dark) ============ */
const Traces: React.FC = () => {
  const frame = useCurrentFrame();
  const rows = 6;
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: 0.35 }}>
      {Array.from({ length: rows }).map((_, r) => {
        const y0 = 160 + r * 150;
        const pts: string[] = [];
        for (let x = 0; x <= 1920; x += 8) {
          const burst = Math.exp(-(((x - ((frame * 14 + r * 260) % 2400) + 240) / 170) ** 2));
          const v = Math.sin(x / 13 + frame / 3 + r) * 6 + (random(`s${r}-${x}-${Math.floor(frame / 2)}`) - 0.5) * (10 + 90 * burst);
          pts.push(`${x},${(y0 + v).toFixed(1)}`);
        }
        return <polyline key={r} points={pts.join(" ")} fill="none" stroke={r % 2 ? "#F0794A" : "#5DB5AE"} strokeWidth={2} />;
      })}
    </svg>
  );
};

export const SemgScene: React.FC = () => {
  const frame = useCurrentFrame();
  const win = interpolate(frame, [8, 34], [0, 1], { ...clamp, easing: ease });
  return (
    <AbsoluteFill>
      <Backdrop dark />
      <Traces />
      <AbsoluteFill style={{ padding: "0 110px", flexDirection: "row", alignItems: "center", gap: 70 }}>
        <div style={{ width: 660 }}>
          <div style={fadeUp(frame, 4)}><Eyebrow color="#86D0C9">Afono sEMG Model</Eyebrow></div>
          <Headline size={82} color="#F3F5F2" style={{ marginTop: 22, ...fadeUp(frame, 10) }}>Reads the muscles of speech.</Headline>
          <div style={{ marginTop: 26, fontFamily: FONT.text, fontSize: 31, lineHeight: 1.4, color: "#B9CBC8", ...fadeUp(frame, 22) }}>
            Mouth the words, no sound. Small skin sensors on the jaw and cheek pick up the signals, and the page's vocabulary narrows what they can mean.
          </div>
          <div style={{ marginTop: 42 }}>
            <Stat light delay={44} value="96.5%" label="word accuracy on a 51-word vocabulary" note="silent speech · one speaker · lab hardware" />
          </div>
        </div>
        <div style={{ opacity: win, transform: `translateY(${(1 - win) * 60}px)` }}>
          <Window width={1000} title="Afono Nullpoint · sEMG Model">
            <OffthreadVideo src={staticFile("nullpoint-decode.mp4")} muted style={{ width: 1000, display: "block" }} />
          </Window>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 8. Whisper mode with ordinary earbuds ============ */
const PHRASES: [string, number][] = [["reply two", 0.998], ["share", 0.001], ["reply one", 0.0004], ["scroll down", 0.0002], ["sign in", 0.0001], ["search", 0.00005]];

export const WhisperScene: React.FC = () => {
  const frame = useCurrentFrame();
  const talking = frame > 40 && frame < 92;
  const decide = interpolate(frame, [100, 130], [0, 1], { ...clamp, easing: ease });
  const sent = interpolate(frame, [150, 166], [0, 1], { ...clamp, easing: ease });
  return (
    <AbsoluteFill>
      <Backdrop tint="#B8DDD8" />
      <AbsoluteFill style={{ padding: "0 130px", flexDirection: "row", alignItems: "center", gap: 90 }}>
        <div style={{ width: 720 }}>
          <div style={fadeUp(frame, 0)}><Eyebrow color={C.signal}>Whisper mode</Eyebrow></div>
          <Headline size={80} style={{ marginTop: 22, ...fadeUp(frame, 6) }}>No sensors yet? Whisper into your earbuds.</Headline>
          <div style={{ marginTop: 28, fontFamily: FONT.text, fontSize: 31, lineHeight: 1.4, color: C.text2, ...fadeUp(frame, 18) }}>
            A speech model on your own computer listens only for what the page allows, so a whisper just has to pick one.
          </div>
          <div style={{ marginTop: 40, display: "flex", gap: 16, flexWrap: "wrap", ...fadeUp(frame, 150) }}>
            {["about 1 s on a laptop CPU", "works offline", "audio never leaves the device"].map((t) => (
              <div key={t} style={{ fontFamily: FONT.mono, fontSize: 21, color: C.teal, padding: "10px 18px", borderRadius: 999, background: "rgba(15,94,99,.08)" }}>{t}</div>
            ))}
          </div>
        </div>
        <Glass style={{ flex: 1, padding: 48, position: "relative" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 26 }}>
            <svg width={70} height={90} viewBox="0 0 70 90">
              <rect x={22} y={34} width={18} height={50} rx={9} fill={C.ink} />
              <ellipse cx={34} cy={30} rx={26} ry={24} fill={C.ink} />
              <circle cx={30} cy={28} r={9} fill="#2E4448" />
            </svg>
            <div style={{ display: "flex", alignItems: "center", gap: 7, height: 90 }}>
              {Array.from({ length: 34 }).map((_, i) => {
                const h = talking ? 10 + 46 * Math.abs(Math.sin(i * 0.9 + frame / 3)) * random(`w${i}-${Math.floor(frame / 3)}`) : 6;
                return <div key={i} style={{ width: 8, height: h, borderRadius: 4, background: talking ? C.signal : "rgba(22,24,26,.18)" }} />;
              })}
            </div>
          </div>
          <div style={{ marginTop: 14, fontFamily: FONT.mono, fontSize: 22, color: C.text3, ...fadeUp(frame, 44) }}>heard: “reply to”</div>
          <div style={{ marginTop: 30, display: "flex", flexDirection: "column", gap: 14 }}>
            {PHRASES.map(([p, prob], i) => {
              const w = interpolate(decide, [0, 1], [1 / PHRASES.length, prob]);
              const win = i === 0;
              return (
                <div key={p} style={{ display: "flex", alignItems: "center", gap: 20, ...fadeUp(frame, 60 + i * 4, 14, 14) }}>
                  <div style={{ width: 220, fontFamily: FONT.text, fontWeight: win && decide > 0.5 ? 700 : 500, fontSize: 30, color: win && decide > 0.5 ? C.teal : C.text2 }}>{p}</div>
                  <div style={{ flex: 1, height: 16, borderRadius: 8, background: "rgba(22,24,26,.07)", overflow: "hidden" }}>
                    <div style={{ width: `${Math.max(0.6, w * 100)}%`, height: "100%", borderRadius: 8, background: win ? C.teal : "rgba(22,24,26,.25)" }} />
                  </div>
                  <div style={{ width: 90, textAlign: "right", fontFamily: FONT.mono, fontSize: 21, color: C.text3 }}>{w.toFixed(3)}</div>
                </div>
              );
            })}
          </div>
          <div style={{ position: "absolute", right: 40, bottom: -34, opacity: sent, transform: `translateY(${(1 - sent) * 20}px)`,
            background: C.ink, color: "#F3F5F2", borderRadius: 16, padding: "16px 26px", fontFamily: FONT.text, fontSize: 28, boxShadow: "0 20px 40px -18px rgba(0,0,0,.4)" }}>
            click reply two <span style={{ color: "#86D0C9" }}>→ sent to the page</span>
          </div>
        </Glass>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 9. How they combine ============ */
export const FusionScene: React.FC = () => {
  const frame = useCurrentFrame();
  const lines: [string, string, string][] = [
    ["The page", "says what's possible.", C.teal2],
    ["Your eyes", "say where.", C.teal],
    ["Your lips", "say which one.", C.signal],
  ];
  const end = fadeUp(frame, 96, 20, 30);
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ justifyContent: "center", padding: "0 200px" }}>
        {lines.map(([a, b, c], i) => (
          <div key={a} style={{ ...fadeUp(frame, 6 + i * 20, 18, 36), fontFamily: FONT.text, fontWeight: 700, fontSize: 96, letterSpacing: "-0.04em", lineHeight: 1.14 }}>
            <span style={{ color: c }}>{a}</span> <span style={{ color: C.text }}>{b}</span>
          </div>
        ))}
        <div style={{ ...end, marginTop: 50, fontFamily: FONT.text, fontSize: 42, color: C.text2 }}>
          Each one covers the others' weak spots. Together they pick the right button.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 10. Nullpoint: the app ============ */
export const AppScene: React.FC = () => {
  const frame = useCurrentFrame();
  const shots = [
    { src: "shot-whisper.png", title: "Whisper", x: -520, r: -6, d: 0 },
    { src: "dom-badges.jpg", title: "DOM Model", x: 0, r: 0, d: 8 },
    { src: "shot-track.png", title: "Track Model", x: 520, r: 6, d: 16 },
  ];
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 110 }}>
        <div style={fadeUp(frame, 0)}><Eyebrow>Afono Nullpoint</Eyebrow></div>
        <Headline size={86} style={{ marginTop: 20, textAlign: "center", ...fadeUp(frame, 6) }}>Runs on your own computer.</Headline>
        <div style={{ marginTop: 18, fontFamily: FONT.text, fontSize: 36, color: C.text2, ...fadeUp(frame, 14) }}>Private by design. Free. Works offline.</div>
      </AbsoluteFill>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-end", paddingBottom: 60 }}>
        <div style={{ position: "relative", width: 1600, height: 560 }}>
          {shots.map((s) => {
            const t = interpolate(frame, [18 + s.d, 46 + s.d], [0, 1], { ...clamp, easing: ease });
            const drift = interpolate(frame, [0, 165], [0, 1]);
            return (
              <div key={s.src} style={{ position: "absolute", left: 800 - 330 + s.x * t, top: 40 + (1 - t) * 200 - (s.x === 0 ? 30 : 0) - drift * 10,
                transform: `rotate(${s.r * t}deg)`, opacity: t, zIndex: s.x === 0 ? 2 : 1 }}>
                <Window width={660} title={s.title}><Img src={staticFile(s.src)} style={{ width: 660, display: "block" }} /></Window>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ 11. Outro ============ */
export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 200 } });
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", textAlign: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 28, transform: `scale(${0.9 + 0.1 * s})`, opacity: s }}>
          <Phi size={170} /><Wordmark size={150} />
        </div>
        <div style={{ marginTop: 40, fontFamily: FONT.text, fontWeight: 600, fontSize: 52, letterSpacing: "-0.02em", color: C.text, ...fadeUp(frame, 14) }}>
          A nonprofit AI lab for hands-free computing.
        </div>
        <div style={{ marginTop: 26, fontFamily: FONT.text, fontSize: 32, color: C.text2, maxWidth: 1300, ...fadeUp(frame, 28) }}>
          Accessible brain–computer interfaces start with a simple human–computer interface.
        </div>
        <div style={{ marginTop: 60, display: "flex", gap: 24, alignItems: "center", ...fadeUp(frame, 46) }}>
          <div style={{ background: C.ink, color: "#fff", borderRadius: 999, padding: "22px 44px", fontFamily: FONT.text, fontWeight: 600, fontSize: 34 }}>Partner with us</div>
          <div style={{ fontFamily: FONT.mono, fontSize: 32, color: C.teal }}>afono.org</div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
