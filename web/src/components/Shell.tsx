import type { ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { useHealth } from "../api/hooks";
import { useMonth } from "../lib/month";
import { monthLabel } from "../lib/format";
import { BrandMark, IconCustomers, IconModel, IconMonitoring, IconOverview, IconPredictions, IconRetention } from "./icons";

const NAV = [
  { to: "/", label: "Overview", icon: IconOverview, end: true },
  { to: "/customers", label: "Customers", icon: IconCustomers },
  { to: "/predictions", label: "Predictions", icon: IconPredictions },
  { to: "/retention", label: "Retention", icon: IconRetention },
  { to: "/performance", label: "Model performance", icon: IconModel },
  { to: "/monitoring", label: "Monitoring", icon: IconMonitoring },
];

/** Keep ?month= when moving between pages. */
export function useMonthHref() {
  const { search } = useLocation();
  const month = new URLSearchParams(search).get("month");
  return (path: string) => (month ? `${path}?month=${month}` : path);
}

export function Shell() {
  const href = useMonthHref();
  const health = useHealth();
  const model = health.data?.model;
  return (
    <div className="app">
      <a href="#main" className="sr-only">
        Skip to content
      </a>
      <aside className="sidebar">
        <NavLink to={href("/")} className="brand" aria-label="ChurnGuard home">
          <span className="brand-mark">
            <BrandMark />
          </span>
          <span className="stack" style={{ ["--gap" as string]: "0" }}>
            <span className="brand-name">ChurnGuard</span>
            <span className="small muted">KKBox retention</span>
          </span>
        </NavLink>
        <nav className="nav" aria-label="Primary">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={href(to)} end={end} className={({ isActive }) => (isActive ? "active" : "")}>
              <Icon />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className="small muted">Serving model</span>
          {model ? (
            <>
              <span className="num" style={{ fontSize: 13 }}>
                {model.name} v{model.version}
              </span>
              <span className="small muted">
                {model.model_type}
                {model.alias ? ` · ${model.alias}` : ""}
              </span>
            </>
          ) : (
            <span className="small muted">{health.isLoading ? "Checking…" : "API offline"}</span>
          )}
        </div>
      </aside>
      <main className="main" id="main">
        <Outlet />
      </main>
    </div>
  );
}

export function HealthPill() {
  const { data, isLoading, isError } = useHealth();
  if (isLoading) return <span className="pill">Checking API…</span>;
  if (isError || data?.status !== "ok")
    return (
      <span className="pill bad" role="status">
        <span className="dot" />
        API offline
      </span>
    );
  return (
    <span className="pill ok" title={`${data.model.name} v${data.model.version}`}>
      <span className="dot" />
      API healthy
    </span>
  );
}

export function MonthSelect() {
  const { month, months, setMonth } = useMonth();
  if (!months.length) return null;
  return (
    <label className="pill" style={{ paddingRight: 4 }}>
      Score month
      <select
        className="select num"
        style={{ height: 26, border: 0, background: "transparent", padding: "0 6px" }}
        value={month}
        onChange={(e) => setMonth(e.target.value)}
      >
        {months.map((m) => (
          <option key={m} value={m}>
            {monthLabel(m)}
          </option>
        ))}
      </select>
    </label>
  );
}

export function PageHead({ title, children, actions, month = true }: { title: ReactNode; children?: ReactNode; actions?: ReactNode; month?: boolean }) {
  return (
    <header className="page-head">
      <div>
        <h1>{title}</h1>
        {children && <p>{children}</p>}
      </div>
      <div className="head-actions">
        {actions}
        {month && <MonthSelect />}
        <HealthPill />
      </div>
    </header>
  );
}
