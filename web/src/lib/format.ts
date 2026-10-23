import type { Tier } from "../api/types";

const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export const MISSING = "—";

const ok = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

export function int(v: number | null | undefined): string {
  return ok(v) ? nf0.format(v) : MISSING;
}

export function pct(v: number | null | undefined, digits = 1): string {
  return ok(v) ? `${(v * 100).toFixed(digits)}%` : MISSING;
}

export function dec(v: number | null | undefined, digits = 3): string {
  return ok(v) ? v.toFixed(digits) : MISSING;
}

/** Compact NT$ amount: NT$6.42M, NT$720k, NT$68.7. Negative values get a real minus sign. */
export function ntd(v: number | null | undefined, opts: { signed?: boolean } = {}): string {
  if (!ok(v)) return MISSING;
  const a = Math.abs(v);
  const body =
    a >= 1e6 ? `${(a / 1e6).toFixed(2)}M` : a >= 1e4 ? `${Math.round(a / 1e3)}k` : a >= 1e3 ? `${(a / 1e3).toFixed(1)}k` : a.toFixed(1);
  const sign = v < 0 ? "−" : opts.signed && v > 0 ? "+" : "";
  return `${sign}NT$${body}`;
}

export function signedPt(delta: number): string {
  const pt = delta * 100;
  return `${pt > 0 ? "+" : pt < 0 ? "−" : ""}${Math.abs(pt).toFixed(2)} pt`;
}

export function monthLabel(m: string): string {
  const [y, mo] = m.split("-").map(Number);
  if (!y || !mo) return m;
  return new Date(Date.UTC(y, mo - 1, 1)).toLocaleString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
}

export function shortMonth(m: string): string {
  const [y, mo] = m.split("-").map(Number);
  if (!y || !mo) return m;
  const name = new Date(Date.UTC(y, mo - 1, 1)).toLocaleString("en-US", { month: "short", timeZone: "UTC" });
  return `${name} ${String(y).slice(2)}`;
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  });
}

export function isoDate(iso: string | null | undefined): string {
  return iso ? iso.slice(0, 10) : MISSING;
}

/** First day of a YYYY-MM score month: the as-of date of that month's decision. */
export function monthStart(m: string): string {
  return `${m}-01`;
}

export function hours(secs: number | null | undefined): string {
  return ok(secs) ? `${(secs / 3600).toFixed(1)} h` : MISSING;
}

export const TIER_LABEL: Record<Tier, string> = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
};

export const TIER_COLOR: Record<Tier, string> = {
  critical: "var(--tier-critical)",
  high: "var(--tier-high)",
  medium: "var(--tier-medium)",
  low: "var(--tier-low)",
};

export function tierOf(p: number): Tier {
  return p >= 0.5 ? "critical" : p >= 0.2 ? "high" : p >= 0.05 ? "medium" : "low";
}

export const CANDIDATE_LABEL: Record<string, string> = {
  logreg: "Logistic regression",
  lgbm: "LightGBM",
  lgbm_robust: "LightGBM robust",
  blend_lgbm_logreg: "Blend LightGBM + logreg",
  blend_robust_logreg: "Blend robust + logreg",
};
