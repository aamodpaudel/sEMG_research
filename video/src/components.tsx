import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { C, FONT } from "./theme";

/* ---------- motion helpers ---------- */
export const useSpring = (delay = 0, config: Parameters<typeof spring>[0]["config"] = { damping: 200 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: frame - delay, fps, config });
};

export const ease = Easing.bezier(0.22, 1, 0.36, 1);

export const fadeUp = (frame: number, delay: number, dur = 18, dist = 28) => {
  const t = interpolate(frame, [delay, delay + dur], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: ease });
  return { opacity: t, transform: `translateY(${(1 - t) * dist}px)` };
};

/* ---------- background: light paper with slow drifting colour fields ---------- */
export const Backdrop: React.FC<{ tint?: string; dark?: boolean }> = ({ tint = C.tealSoft, dark = false }) => {
  const frame = useCurrentFrame();
  const a = frame / 90;
  if (dark) {
    return (
      <AbsoluteFill style={{ background: "#0B1719" }}>
        <AbsoluteFill style={{ background: `radial-gradient(900px 700px at ${50 + 12 * Math.sin(a)}% ${40 + 8 * Math.cos(a)}%, rgba(93,181,174,0.22), transparent 70%)` }} />
        <AbsoluteFill style={{ background: `radial-gradient(700px 600px at ${80 - 10 * Math.cos(a * 0.8)}% 80%, rgba(240,121,74,0.14), transparent 70%)` }} />
      </AbsoluteFill>
    );
  }
  return (
    <AbsoluteFill style={{ background: C.paper }}>
      <AbsoluteFill style={{ background: `radial-gradient(1100px 800px at ${18 + 10 * Math.sin(a)}% ${22 + 6 * Math.cos(a)}%, ${tint}55, transparent 70%)` }} />
      <AbsoluteFill style={{ background: `radial-gradient(900px 800px at ${86 - 8 * Math.cos(a * 0.7)}% ${78 + 6 * Math.sin(a * 0.9)}%, rgba(217,85,43,0.16), transparent 70%)` }} />
    </AbsoluteFill>
  );
};

/* ---------- the φ mark: ring drawn, bar dropped (BRAND.md motion) ---------- */
export const Phi: React.FC<{ size: number; ringDraw?: number; barDrop?: number; squash?: number; light?: boolean }> = ({
  size, ringDraw = 1, barDrop = 1, squash = 0, light = false,
}) => {
  const circ = 2 * Math.PI * 22;
  const ring = light ? C.tealSoft : C.teal;
  const barY = interpolate(barDrop, [0, 1], [-70, 0]);
  return (
    <svg width={size * (64 / 96)} height={size} viewBox="0 0 64 96" style={{ overflow: "visible", transform: `scale(${1 + 0.06 * squash}, ${1 - 0.06 * squash})` }}>
      <circle cx={32} cy={48} r={22} fill="none" stroke={ring} strokeWidth={6.5} strokeDasharray={circ}
        strokeDashoffset={circ * (1 - ringDraw)} strokeLinecap="round" transform="rotate(-90 32 48)" />
      <line x1={32} y1={8 + barY} x2={32} y2={88 + barY} stroke={light ? "#F0794A" : C.signal} strokeWidth={6.5}
        strokeLinecap="round" opacity={barDrop > 0.01 ? 1 : 0} />
    </svg>
  );
};

export const Wordmark: React.FC<{ size: number; reveal?: number; light?: boolean }> = ({ size, reveal = 1, light = false }) => {
  const letters = "afono".split("");
  return (
    <span style={{ fontFamily: FONT.mark, fontWeight: 800, fontSize: size, letterSpacing: "-0.045em", color: light ? "#ECF3F1" : C.teal, display: "inline-flex" }}>
      {letters.map((l, i) => {
        const t = interpolate(reveal * letters.length - i, [0, 1], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
        return <span key={i} style={{ opacity: t, transform: `translateY(${(1 - t) * size * 0.35}px)`, display: "inline-block" }}>{l}</span>;
      })}
    </span>
  );
};

/* ---------- text pieces ---------- */
export const Eyebrow: React.FC<{ children: React.ReactNode; color?: string; style?: React.CSSProperties }> = ({ children, color = C.teal, style }) => (
  <div style={{ fontFamily: FONT.mono, fontSize: 24, letterSpacing: "0.18em", textTransform: "uppercase", color, ...style }}>{children}</div>
);

export const Headline: React.FC<{ children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties }> = ({ children, size = 96, color = C.text, style }) => (
  <div style={{ fontFamily: FONT.text, fontWeight: 700, fontSize: size, lineHeight: 1.04, letterSpacing: "-0.035em", color, ...style }}>{children}</div>
);

/** Words rise in one after another. */
export const WordStagger: React.FC<{ text: string; start: number; size?: number; color?: string; accent?: string[]; per?: number; style?: React.CSSProperties }> = ({
  text, start, size = 110, color = C.text, accent = [], per = 4, style,
}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ fontFamily: FONT.text, fontWeight: 700, fontSize: size, lineHeight: 1.05, letterSpacing: "-0.04em", color, ...style }}>
      {text.split(" ").map((w, i) => {
        const s = fadeUp(frame, start + i * per, 16, size * 0.4);
        const hot = accent.includes(w.replace(/[.,]/g, ""));
        return <span key={i} style={{ display: "inline-block", marginRight: "0.24em", ...s, color: hot ? C.teal : undefined }}>{w}</span>;
      })}
    </div>
  );
};

/** A measured number, always with its conditions. */
export const Stat: React.FC<{ value: string; label: string; note?: string; delay: number; light?: boolean }> = ({ value, label, note, delay, light = false }) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ ...fadeUp(frame, delay), display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ fontFamily: FONT.text, fontWeight: 700, fontSize: 84, letterSpacing: "-0.04em", color: light ? "#86D0C9" : C.teal, lineHeight: 1 }}>{value}</div>
      <div style={{ fontFamily: FONT.text, fontWeight: 500, fontSize: 28, color: light ? "#ECF3F1" : C.text }}>{label}</div>
      {note ? <div style={{ fontFamily: FONT.mono, fontSize: 19, color: light ? "#9DB4B0" : C.text3 }}>{note}</div> : null}
    </div>
  );
};

export const Glass: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{
    background: C.glass, borderRadius: 36, border: "1px solid rgba(255,255,255,0.85)",
    boxShadow: "0 1px 0 rgba(255,255,255,.95) inset, 0 0 0 1px rgba(22,24,26,.05), 0 40px 90px -40px rgba(15,40,45,.35)",
    backdropFilter: "blur(28px) saturate(170%)", ...style,
  }}>{children}</div>
);

/** A browser/app window chrome around footage. */
export const Window: React.FC<{ children: React.ReactNode; width: number; title?: string; style?: React.CSSProperties }> = ({ children, width, title, style }) => (
  <div style={{ width, borderRadius: 22, overflow: "hidden", background: C.white, boxShadow: "0 0 0 1px rgba(22,24,26,.08), 0 50px 120px -40px rgba(15,40,45,.45)", ...style }}>
    <div style={{ height: 46, display: "flex", alignItems: "center", gap: 9, padding: "0 18px", background: "#F2F4F4", borderBottom: `1px solid ${C.line}` }}>
      {["#FF5F57", "#FEBC2E", "#28C840"].map((c) => <div key={c} style={{ width: 13, height: 13, borderRadius: 7, background: c }} />)}
      {title ? <div style={{ marginLeft: 16, fontFamily: FONT.mono, fontSize: 17, color: C.text3 }}>{title}</div> : null}
    </div>
    {children}
  </div>
);

export const Chip: React.FC<{ children: React.ReactNode; delay: number; color?: string }> = ({ children, delay, color = C.teal }) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ ...fadeUp(frame, delay, 16, 18), display: "inline-flex", alignItems: "center", gap: 12, padding: "16px 28px", borderRadius: 999,
      background: C.white, boxShadow: "0 0 0 1px rgba(22,24,26,.07), 0 18px 40px -24px rgba(15,40,45,.35)",
      fontFamily: FONT.text, fontWeight: 600, fontSize: 34, color: C.text }}>
      <span style={{ width: 12, height: 12, borderRadius: 6, background: color }} />{children}
    </div>
  );
};
