import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";

import { api, ApiError } from "../api/client";
import { useHistory, useOverview, useRegistry } from "../api/hooks";
import type { ScoreResponse } from "../api/types";
import { Histogram } from "../components/charts";
import { PageHead, useMonthHref } from "../components/Shell";
import { Card, CardSkeleton, EmptyState, ErrorState, Kpi, KpiSkeletons, Segmented, Skeleton } from "../components/ui";
import { featureLabel } from "../lib/features";
import { dateTime, int, monthLabel, monthStart, ntd, pct, tierOf, TIER_COLOR } from "../lib/format";
import { useMonth } from "../lib/month";

export function PredictionsPage() {
  const { month } = useMonth();
  const overview = useOverview(month);
  const [scale, setScale] = useState<"log" | "linear">("log");
  const o = overview.data;
  const below = o?.histogram[0];
  const bins = o?.histogram.slice(1) ?? [];

  return (
    <>
      <PageHead title="Predictions">Batch scores, the score distribution, model versions and on-demand scoring through the API</PageHead>

      {overview.isError ? (
        <Card>
          <ErrorState error={overview.error} what="Scores" retry={() => overview.refetch()} />
        </Card>
      ) : !o ? (
        <KpiSkeletons n={4} />
      ) : (
        <section className="grid" style={{ ["--min" as string]: "180px" }}>
          <Kpi label="Batch" value={o.month} sub={dateTime(o.scored_at)} />
          <Kpi label="Rows scored" value={int(o.members)} sub={`${o.model.name} v${o.model.version} · ${o.model.model_type}`} delay={40} />
          <Kpi label="Mean / median p" value={`${pct(o.mean_p, 2)} / ${pct(o.median_p, 2)}`} sub="a long right tail" delay={80} />
          <Kpi label="Flagged to contact" value={int(o.contacted)} sub="positive expected value" delay={120} />
        </section>
      )}

      <section className="row">
        <Card
          className="grow"
          title={o ? `Score distribution · ${monthLabel(o.month)}` : "Score distribution"}
          action={<Segmented label="Scale" value={scale} onChange={setScale} options={[{ value: "log", label: "Log" }, { value: "linear", label: "Linear" }]} />}
        >
          {!o ? (
            <Skeleton h={200} />
          ) : (
            <div style={{ display: "flex", gap: 14, alignItems: "stretch", flexWrap: "wrap" }}>
              {below && (
                <div className="well" style={{ flex: "0 0 132px", justifyContent: "flex-end" }}>
                  <span className="small muted">Below {pct(below.hi, 0)}</span>
                  <span className="num" style={{ fontSize: 18 }}>{int(below.members)}</span>
                  <span className="small muted">{pct(below.members / o.members)} of members, kept off the chart so the tail is visible</span>
                </div>
              )}
              <div style={{ flex: "1 1 320px", minWidth: 0 }}>
                <Histogram
                  bins={bins}
                  log={scale === "log"}
                  color={(lo) => TIER_COLOR[tierOf(lo + 1e-9)]}
                  format={(b) => `${pct(b.lo, 0)}–${pct(b.hi, 0)}: ${int(b.members)} members`}
                />
              </div>
            </div>
          )}
          <div className="legend">
            <span><span className="swatch" style={{ background: "var(--tier-medium)" }} />Medium 5–20%</span>
            <span><span className="swatch" style={{ background: "var(--tier-high)" }} />High 20–50%</span>
            <span><span className="swatch" style={{ background: "var(--tier-critical)" }} />Critical ≥50%</span>
          </div>
        </Card>
        <ScoreMember month={month} />
      </section>

      <section className="grid" style={{ ["--min" as string]: "420px" }}>
        <RegistryCard />
        <HistoryCard />
      </section>
    </>
  );
}

function ScoreMember({ month }: { month?: string }) {
  const href = useMonthHref();
  const [id, setId] = useState("");
  const [asOf, setAsOf] = useState(month ? monthStart(month) : "");
  // The score month arrives after the first render; default the date to its decision date.
  useEffect(() => {
    if (month) setAsOf((cur) => cur || monthStart(month));
  }, [month]);
  const [state, setState] = useState<
    { kind: "idle" } | { kind: "invalid" } | { kind: "loading" } | { kind: "done"; r: ScoreResponse } | { kind: "error"; e: ApiError }
  >({ kind: "idle" });

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!/^\d+$/.test(id.trim()) || !asOf) {
      setState({ kind: "invalid" });
      return;
    }
    setState({ kind: "loading" });
    try {
      setState({ kind: "done", r: await api.scoreMember(Number(id.trim()), asOf) });
    } catch (err) {
      setState({ kind: "error", e: err instanceof ApiError ? err : new ApiError(0, String(err)) });
    }
  };

  return (
    <Card className="side" title="Score a member" action={<code className="inline">GET /members/&#123;id&#125;/score</code>}>
      <form onSubmit={submit} className="stack" noValidate>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <label className="field" style={{ flex: "1 1 120px" }}>
            Member ID
            <input
              className="input num"
              inputMode="numeric"
              placeholder="e.g. 1184204"
              value={id}
              onChange={(e) => setId(e.target.value)}
              aria-invalid={state.kind === "invalid"}
              aria-describedby="score-err"
            />
          </label>
          <label className="field" style={{ flex: "1 1 150px" }}>
            As of
            <input className="input num" type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} required />
          </label>
        </div>
        <span id="score-err" role="alert" className="small warn" style={{ minHeight: 0 }}>
          {state.kind === "invalid" ? "Enter a whole-number member ID and a date." : ""}
        </span>
        <button className="btn primary" type="submit" disabled={state.kind === "loading"}>
          {state.kind === "loading" ? "Scoring…" : "Score member"}
        </button>
      </form>

      <div aria-live="polite">
        {state.kind === "loading" && (
          <div className="well" aria-busy="true">
            <Skeleton h={26} w="40%" />
            <Skeleton h={12} w="90%" />
            <Skeleton h={12} w="70%" />
          </div>
        )}
        {state.kind === "done" && (
          <div className="well" style={{ gap: 10 }}>
            <div className="between">
              <span className="num" style={{ fontSize: 24 }}>{pct(state.r.p_churn)}</span>
              <span className="small pos" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <span className="dot" />
                200 · model v{state.r.model.version}
              </span>
            </div>
            <div className="between small">
              <span className="muted">Decision</span>
              <span>
                {state.r.contact ? "Contact" : "Hold"} · EV <span className="num">{ntd(state.r.expected_value, { signed: true })}</span>
              </span>
            </div>
            {state.r.reasons[0] && (
              <div className="between small">
                <span className="muted">Top reason</span>
                <span>{featureLabel(state.r.reasons[0].feature)}</span>
              </div>
            )}
            {month && state.r.member_id !== null && asOf === monthStart(month) && (
              <Link to={href(`/customers/${state.r.member_id}`)} className="small">
                Open full profile
              </Link>
            )}
          </div>
        )}
        {state.kind === "error" && (
          <div className="dashed" role="alert">
            <strong style={{ color: "var(--text)", fontWeight: 500 }}>
              {state.e.status === 404 ? "No history for this member" : state.e.status === 0 ? "API unreachable" : `Request failed (HTTP ${state.e.status})`}
            </strong>
            <div>{state.e.detail}</div>
          </div>
        )}
      </div>
      <p className="card-note">
        Batch scoring is a CLI job, not an API call: <code className="inline">make score MONTH=YYYY-MM</code>
      </p>
    </Card>
  );
}

function RegistryCard() {
  const { data, error, isLoading } = useRegistry();
  return (
    <Card flush title="Model registry" action={<span className="small muted num">MLflow</span>}>
      {isLoading ? (
        <div style={{ padding: 16 }}>
          <Skeleton h={120} />
        </div>
      ) : error || !data ? (
        <ErrorState error={error} what="Model registry" />
      ) : data.versions.length === 0 ? (
        <EmptyState title="No registered models">Run <code>make select</code> to register a champion.</EmptyState>
      ) : (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Type</th>
                <th scope="col">Aliases</th>
                <th scope="col">Run</th>
              </tr>
            </thead>
            <tbody>
              {data.versions.map((v) => (
                <tr key={`${v.name}-${v.version}`} className={v.aliases.includes("champion") ? "highlight" : ""}>
                  <td className="num">
                    {v.name} v{v.version}
                  </td>
                  <td>{v.model_type ?? "—"}</td>
                  <td>
                    <span style={{ display: "inline-flex", gap: 6, flexWrap: "wrap" }}>
                      {v.aliases.length ? v.aliases.map((a) => <span key={a} className={`badge ${a === "champion" ? "violet" : ""}`}>{a}</span>) : <span className="muted small">none</span>}
                    </span>
                  </td>
                  <td className="num muted">{v.run_id.slice(0, 8)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="card-note">Serving loads the version behind the <code className="inline">champion</code> alias. A challenger is promoted only if it passes gate v2.</p>
    </Card>
  );
}

function HistoryCard() {
  const { data, error, isLoading } = useHistory();
  if (isLoading) return <CardSkeleton />;
  const served = new Map((data?.served ?? []).map((s) => [s.month, s]));
  const replay = new Map((data?.replay ?? []).map((r) => [r.month, r]));
  const months = [...new Set([...served.keys(), ...replay.keys()])].sort().reverse();
  return (
    <Card flush title="Scoring history" action={<span className="small muted">served and replayed months</span>}>
      {error || !data ? (
        <ErrorState error={error} what="History" />
      ) : (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Month</th>
                <th scope="col">Source</th>
                <th scope="col" className="r">Members</th>
                <th scope="col" className="r">Contacted</th>
                <th scope="col" className="r">Mean p</th>
              </tr>
            </thead>
            <tbody>
              {months.map((m) => {
                const s = served.get(m);
                const r = replay.get(m);
                return (
                  <tr key={m}>
                    <td className="num">{m}</td>
                    <td style={{ color: s ? "var(--violet-text)" : "var(--text-muted)" }}>{s ? "served" : "replayed"}</td>
                    <td className="r num">{s ? int(s.members) : <span className="faint">not stored</span>}</td>
                    <td className="r num">{s ? int(s.contacted) : <span className="faint">—</span>}</td>
                    <td className="r num">{pct(s?.mean_p ?? r?.mean_p, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
