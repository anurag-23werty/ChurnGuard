import { useId, useState, type ReactNode } from "react";

// Small, dependency-free SVG/HTML charts. Each takes already-aggregated API data and
// exposes its values to assistive tech (aria-label or a text readout).

export interface Series {
  name: string;
  color: string;
  values: (number | null)[];
  dashed?: boolean;
  width?: number;
  /** Points before this index are drawn hollow (e.g. replayed vs served). */
  hollowBefore?: number;
}

function domain(values: number[]): [number, number] {
  if (!values.length) return [0, 1];
  const a = Math.min(...values);
  const b = Math.max(...values);
  const p = (b - a) * 0.12 || Math.abs(b) * 0.1 || 0.01;
  return [a - p, b + p];
}

function niceTicks(lo: number, hi: number, n = 4): number[] {
  const step = (hi - lo) / n;
  return Array.from({ length: n + 1 }, (_, i) => lo + step * i);
}

export function LineChart({
  series,
  labels,
  format,
  height = 220,
  band,
  ariaLabel,
  showValues = false,
}: {
  series: Series[];
  labels: string[];
  format: (v: number) => string;
  height?: number;
  band?: { index: number; label: string };
  ariaLabel: string;
  showValues?: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 640;
  const H = height;
  const pad = { l: 56, r: 20, t: 24, b: 30 };
  const n = labels.length;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null && Number.isFinite(v)));
  const [lo, hi] = domain(all);
  const x = (i: number) => pad.l + (n === 1 ? (W - pad.l - pad.r) / 2 : (i * (W - pad.l - pad.r)) / (n - 1));
  const y = (v: number) => pad.t + ((hi - v) / (hi - lo)) * (H - pad.t - pad.b);
  const ticks = niceTicks(lo, hi);
  const step = n > 1 ? (W - pad.l - pad.r) / (n - 1) : W;

  return (
    <div className="chart" style={{ position: "relative" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={ariaLabel}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = ((e.clientX - r.left) / r.width) * W;
          const i = Math.round((px - pad.l) / step);
          setHover(i >= 0 && i < n ? i : null);
        }}
      >
        {band && band.index < n && (
          <g>
            <rect x={x(band.index) - step / 2} y={pad.t - 10} width={step} height={H - pad.t - pad.b + 10} fill="var(--amber-bg)" opacity={0.55} />
            <text x={x(band.index)} y={pad.t - 14} textAnchor="middle" style={{ fill: "var(--amber-text)", fontFamily: "var(--font)" }}>
              {band.label}
            </text>
          </g>
        )}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="#1f222a" />
            <text x={pad.l - 8} y={y(t) + 4} textAnchor="end">
              {format(t)}
            </text>
          </g>
        ))}
        {labels.map((l, i) => (
          <text key={l} x={x(i)} y={H - 8} textAnchor="middle" style={{ fontFamily: "var(--font)" }}>
            {l}
          </text>
        ))}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={H - pad.b} stroke="var(--line-strong)" />}
        {series.map((s) => {
          const pts = s.values.map((v, i) => (v === null ? null : ([x(i), y(v)] as const)));
          const d = pts
            .map((p, i) => (p ? `${i === 0 || !pts[i - 1] ? "M" : "L"}${p[0].toFixed(1)} ${p[1].toFixed(1)}` : ""))
            .join(" ");
          return (
            <g key={s.name}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" />
              {pts.map((p, i) =>
                p ? (
                  <circle
                    key={i}
                    cx={p[0]}
                    cy={p[1]}
                    r={hover === i ? 5 : 3.5}
                    fill={s.hollowBefore !== undefined && i < s.hollowBefore ? "var(--surface)" : s.color}
                    stroke={s.color}
                    strokeWidth={2}
                  />
                ) : null,
              )}
              {showValues &&
                pts.map((p, i) =>
                  p ? (
                    <text key={`v${i}`} x={p[0]} y={p[1] - 10} textAnchor="middle" style={{ fill: "var(--text)" }}>
                      {format(s.values[i] as number)}
                    </text>
                  ) : null,
                )}
            </g>
          );
        })}
      </svg>
      {hover !== null && (
        <div
          className="tip-box"
          style={{
            opacity: 1,
            transform: "none",
            width: "auto",
            minWidth: 160,
            right: "auto",
            left: `${(x(hover) / W) * 100}%`,
            top: 0,
            bottom: "auto",
            translate: hover > n / 2 ? "-105% 0" : "8px 0",
          }}
          aria-hidden="true"
        >
          <div className="faint small" style={{ marginBottom: 4 }}>
            {labels[hover]}
          </div>
          {series.map((s) =>
            s.values[hover] === null || s.values[hover] === undefined ? null : (
              <div key={s.name} className="between" style={{ gap: 12, alignItems: "center" }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span className="swatch" style={{ background: s.color }} />
                  {s.name}
                </span>
                <span className="num">{format(s.values[hover] as number)}</span>
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}

export function Histogram({
  bins,
  color,
  format,
  log = false,
}: {
  bins: { lo: number; hi: number; members: number }[];
  color: (lo: number) => string;
  format: (b: { lo: number; hi: number; members: number }) => string;
  log?: boolean;
}) {
  const [active, setActive] = useState<number | null>(null);
  const max = Math.max(1, ...bins.map((b) => b.members));
  const h = (v: number) => {
    if (v <= 0) return 0;
    if (!log) return (v / max) * 100;
    const floor = Math.log10(Math.max(1, Math.min(...bins.filter((b) => b.members > 0).map((b) => b.members)))) - 0.3;
    return ((Math.log10(v) - floor) / (Math.log10(max) - floor)) * 100;
  };
  return (
    <div className="stack" style={{ ["--gap" as string]: "6px" }}>
      <div className="hist" role="group" aria-label="Score distribution bins">
        {bins.map((b, i) => (
          <button
            key={b.lo}
            type="button"
            aria-label={format(b)}
            title={format(b)}
            onMouseEnter={() => setActive(i)}
            onFocus={() => setActive(i)}
            onMouseLeave={() => setActive(null)}
            onBlur={() => setActive(null)}
            style={{ height: `${Math.max(h(b.members), b.members ? 1.5 : 0)}%`, background: color(b.lo) }}
          />
        ))}
      </div>
      <div className="between num small faint">
        <span>{Math.round(bins[0].lo * 100)}%</span>
        <span>{Math.round(((bins[0].lo + bins[bins.length - 1].hi) / 2) * 100)}%</span>
        <span>{Math.round(bins[bins.length - 1].hi * 100)}%</span>
      </div>
      <div className="small muted num" aria-live="polite" style={{ minHeight: 18 }}>
        {active !== null ? format(bins[active]) : "Hover or focus a bar for its count"}
      </div>
    </div>
  );
}

export function DivergingRow({
  label,
  value,
  max,
  color,
  format,
}: {
  label: ReactNode;
  value: number;
  max: number;
  color: string;
  format: (v: number) => string;
}) {
  const w = `${Math.min(100, (Math.abs(value) / max) * 100)}%`;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "72px minmax(0,1fr) minmax(0,1fr) 84px", gap: 8, alignItems: "center", fontSize: 12 }}>
      <span>{label}</span>
      <div style={{ display: "flex", justifyContent: "flex-end", height: 10 }}>
        {value < 0 && <div style={{ width: w, background: color, borderRadius: "3px 0 0 3px" }} />}
      </div>
      <div style={{ borderLeft: "1px solid var(--line-strong)", height: 10, display: "flex" }}>
        {value > 0 && <div style={{ width: w, background: color, borderRadius: "0 3px 3px 0" }} />}
      </div>
      <span className={`num ${value < 0 ? "muted" : ""}`} style={{ textAlign: "right" }}>
        {format(value)}
      </span>
    </div>
  );
}

export function Gauge({ value, label, sub, color }: { value: number; label: string; sub?: string; color: string }) {
  const v = Math.max(0, Math.min(1, value));
  const a = Math.PI * (1 - v);
  const ex = 100 + 80 * Math.cos(a);
  const ey = 100 - 80 * Math.sin(a);
  const id = useId();
  return (
    <svg viewBox="0 0 200 118" width="240" style={{ maxWidth: "100%" }} role="img" aria-labelledby={id}>
      <title id={id}>{`${label}${sub ? `, ${sub}` : ""}`}</title>
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="var(--track)" strokeWidth="12" strokeLinecap="round" />
      {v > 0 && (
        <path d={`M20 100 A80 80 0 0 1 ${ex.toFixed(1)} ${ey.toFixed(1)}`} fill="none" stroke={color} strokeWidth="12" strokeLinecap="round" />
      )}
      <text x="100" y="92" textAnchor="middle" fill="var(--text)" fontSize="30" fontFamily="var(--mono)">
        {label}
      </text>
      {sub && (
        <text x="100" y="113" textAnchor="middle" fill="var(--text-muted)" fontSize="11" fontFamily="var(--font)">
          {sub}
        </text>
      )}
    </svg>
  );
}

export function StackedBar({ parts, ariaLabel }: { parts: { key: string; value: number; color: string; title: string }[]; ariaLabel: string }) {
  const total = parts.reduce((s, p) => s + p.value, 0) || 1;
  return (
    <div role="img" aria-label={ariaLabel} style={{ display: "flex", height: 36, borderRadius: 8, overflow: "hidden", gap: 2 }}>
      {parts.map((p) => (
        <div key={p.key} title={p.title} style={{ width: `${(p.value / total) * 100}%`, background: p.color, transition: "width .5s var(--ease)" }} />
      ))}
    </div>
  );
}
