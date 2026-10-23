import { useEffect, useId, useRef, type CSSProperties, type ReactNode } from "react";

import { ApiError } from "../api/client";
import type { Tier } from "../api/types";
import { TIER_LABEL } from "../lib/format";
import { IconAlert, IconEmpty } from "./icons";

export function Card({
  title,
  action,
  children,
  flush,
  className = "",
  style,
  note,
  labelledBy,
}: {
  title?: ReactNode;
  action?: ReactNode;
  children?: ReactNode;
  flush?: boolean;
  className?: string;
  style?: CSSProperties;
  note?: ReactNode;
  labelledBy?: string;
}) {
  const id = useId();
  return (
    <section className={`card ${flush ? "flush" : ""} ${className}`} style={style} aria-labelledby={title ? (labelledBy ?? id) : undefined}>
      {(title || action) && (
        <div className="card-head">
          {title && (
            <h2 className="card-title" id={id}>
              {title}
            </h2>
          )}
          {action}
        </div>
      )}
      {children}
      {note && <p className="card-note">{note}</p>}
    </section>
  );
}

export function Kpi({
  label,
  value,
  sub,
  tip,
  delay = 0,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tip?: string;
  delay?: number;
}) {
  return (
    <div className="card kpi rise" style={{ animationDelay: `${delay}ms` }}>
      <span className="kpi-label">
        {label}
        {tip && <InfoTip label={`About ${label.toLowerCase()}`}>{tip}</InfoTip>}
      </span>
      <span className="kpi-value">{value}</span>
      {sub && <span className="kpi-sub">{sub}</span>}
    </div>
  );
}

export function Skeleton({ h = 14, w = "100%", style }: { h?: number; w?: number | string; style?: CSSProperties }) {
  return <span className="skeleton" style={{ display: "block", height: h, width: w, ...style }} aria-hidden="true" />;
}

export function KpiSkeletons({ n = 5 }: { n?: number }) {
  return (
    <div className="grid" style={{ ["--min" as string]: "200px" }} aria-busy="true" aria-label="Loading figures">
      {Array.from({ length: n }, (_, i) => (
        <div key={i} className="card kpi">
          <Skeleton h={12} w="50%" />
          <Skeleton h={26} w="70%" />
          <Skeleton h={12} w="80%" />
        </div>
      ))}
    </div>
  );
}

export function CardSkeleton({ h = 220 }: { h?: number }) {
  return (
    <div className="card" aria-busy="true" aria-label="Loading">
      <Skeleton h={14} w="35%" />
      <Skeleton h={h} />
    </div>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <IconEmpty color="var(--text-faint)" />
      <span className="empty-title">{title}</span>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

/** Error panel. 404/503 mean "this artifact has not been produced": shown as unavailable,
 *  with the backend's hint (which names the make target), never as fake data. */
export function ErrorState({ error, what, retry }: { error: unknown; what: string; retry?: () => void }) {
  const e = error instanceof ApiError ? error : null;
  const unavailable = e?.unavailable;
  return (
    <div className="empty" role={unavailable ? "status" : "alert"}>
      <IconAlert color={unavailable ? "var(--text-faint)" : "var(--amber)"} width={28} height={28} />
      <span className="empty-title">{unavailable ? `${what} unavailable` : `Could not load ${what.toLowerCase()}`}</span>
      <p>
        {e ? (
          <>
            {e.status ? <span className="num">HTTP {e.status} · </span> : null}
            {e.detail}
          </>
        ) : (
          String((error as Error)?.message ?? error)
        )}
      </p>
      {retry && !unavailable && (
        <button className="btn" type="button" onClick={retry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function InfoTip({ label, children }: { label: string; children: ReactNode }) {
  const id = useId();
  return (
    <span className="tip">
      <button type="button" className="tip-btn" aria-label={label} aria-describedby={id}>
        ?
      </button>
      <span className="tip-box" role="tooltip" id={id}>
        {children}
      </span>
    </span>
  );
}

export function TierBadge({ tier }: { tier: Tier }) {
  return <span className={`badge ${tier}`}>{TIER_LABEL[tier]}</span>;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Track({ value, color, height = 8, label }: { value: number; color: string; height?: number; label?: string }) {
  const w = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className="track"
      style={{ height }}
      role={label ? "meter" : undefined}
      aria-label={label}
      aria-valuemin={label ? 0 : undefined}
      aria-valuemax={label ? 100 : undefined}
      aria-valuenow={label ? Math.round(w) : undefined}
    >
      <span style={{ width: `${w}%`, minWidth: w > 0 ? 3 : 0, background: color }} />
    </div>
  );
}

export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  children: ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} className="confirm" onCancel={onCancel} onClose={onCancel} aria-labelledby="confirm-title">
      <h2 id="confirm-title">{title}</h2>
      <p>{children}</p>
      <div className="actions">
        <button type="button" className="btn ghost" onClick={onCancel}>
          Cancel
        </button>
        <button type="button" className="btn primary" onClick={onConfirm} autoFocus>
          {confirmLabel}
        </button>
      </div>
    </dialog>
  );
}
