import { Link } from "react-router-dom";

import { useBacktest, useHistory, useMonitoring, useOverview, useSelection } from "../api/hooks";
import type { Tier } from "../api/types";
import { DivergingRow, LineChart, StackedBar } from "../components/charts";
import { PageHead, useMonthHref } from "../components/Shell";
import { Card, CardSkeleton, EmptyState, ErrorState, Kpi, KpiSkeletons, Track } from "../components/ui";
import { featureLabel } from "../lib/features";
import { dateTime, dec, int, monthLabel, ntd, pct, shortMonth, signedPt, TIER_COLOR, TIER_LABEL } from "../lib/format";
import { useMonth } from "../lib/month";

export function OverviewPage() {
  const { month, isLoading: monthsLoading, error: monthsError } = useMonth();
  const overview = useOverview(month);
  const history = useHistory();
  const selection = useSelection();
  const href = useMonthHref();
  const o = overview.data;

  // Month-over-month change from the served months (or the replay for earlier ones).
  const trend = (() => {
    const served = new Map((history.data?.served ?? []).map((s) => [s.month, s.mean_p]));
    const replay = new Map((history.data?.replay ?? []).map((r) => [r.month, r.mean_p]));
    const months = [...new Set([...replay.keys(), ...served.keys()])].sort();
    return months.map((m) => ({ month: m, mean_p: served.get(m) ?? replay.get(m) ?? null, served: served.has(m) }));
  })();
  const idx = o ? trend.findIndex((t) => t.month === o.month) : -1;
  const prev = idx > 0 ? trend[idx - 1] : undefined;

  const champion = selection.data?.summary.find((s) => s.candidate === selection.data?.selected);
  const worstMonth = selection.data?.per_month
    .filter((r) => r.candidate === selection.data?.selected)
    .reduce<{ month: string; roc_auc: number } | null>((w, r) => (!w || r.roc_auc < w.roc_auc ? r : w), null);

  if (monthsError) {
    return (
      <>
        <PageHead title="Overview" />
        <Card>
          <ErrorState error={monthsError} what="Scores" />
        </Card>
      </>
    );
  }
  if (!monthsLoading && !month) {
    return (
      <>
        <PageHead title="Overview" />
        <Card>
          <EmptyState title="No scored months yet">
            Run <code>make score</code> to batch-score the members due to renew, then reload.
          </EmptyState>
        </Card>
      </>
    );
  }

  const highShare = o ? (o.tiers.find((t) => t.tier === "high")!.share + o.tiers.find((t) => t.tier === "critical")!.share) : 0;
  const highRisk = o
    ? (o.tiers.find((t) => t.tier === "high")!.revenue_at_risk + o.tiers.find((t) => t.tier === "critical")!.revenue_at_risk) / o.revenue_at_risk
    : 0;
  const highMembers = o ? o.tiers.filter((t) => t.tier === "high" || t.tier === "critical").reduce((s, t) => s + t.members, 0) : 0;

  return (
    <>
      <PageHead title="Overview">
        {o ? (
          <>
            {monthLabel(o.month)} scoring run · members whose plan expires this month · batch scored {dateTime(o.scored_at)}
          </>
        ) : (
          "Loading the latest scoring run…"
        )}
      </PageHead>

      {overview.isError ? (
        <Card>
          <ErrorState error={overview.error} what="Overview" retry={() => overview.refetch()} />
        </Card>
      ) : !o ? (
        <KpiSkeletons />
      ) : (
        <section className="grid" style={{ ["--min" as string]: "200px" }} aria-label="Key figures">
          <Kpi label="Members scored" value={int(o.members)} sub={`due to renew in ${monthLabel(o.month)}`} />
          <Kpi
            label="Predicted churn rate"
            value={pct(o.mean_p, 2)}
            delay={40}
            sub={
              prev?.mean_p != null ? (
                <>
                  mean probability · <span className={o.mean_p <= prev.mean_p ? "pos num" : "warn num"}>{signedPt(o.mean_p - prev.mean_p)}</span> vs{" "}
                  {shortMonth(prev.month)}
                </>
              ) : (
                `median ${pct(o.median_p, 2)}`
              )
            }
          />
          <Kpi
            label="Revenue at risk"
            value={ntd(o.revenue_at_risk)}
            sub="expected monthly revenue lost"
            tip="Sum over members of churn probability × plan-normalised monthly revenue."
            delay={80}
          />
          <Kpi
            label="Contact list"
            value={int(o.contacted)}
            delay={120}
            sub={
              <>
                {pct(o.contact_share, 2)} of members · EV <span className="pos num">{ntd(o.expected_net_value, { signed: true })}</span>
              </>
            }
            tip="Members whose expected value of a retention offer is positive: p × save rate × CLV − offer cost > 0."
          />
          {champion ? (
            <Kpi
              label="Model ROC AUC"
              value={dec(champion.mean_roc_auc)}
              delay={160}
              sub={
                <>
                  mean out-of-time · worst <span className="num">{dec(champion.worst_roc_auc)}</span>
                  {worstMonth ? ` (${shortMonth(worstMonth.month)})` : ""}
                </>
              }
            />
          ) : (
            <Kpi label="Model ROC AUC" value="—" sub={selection.isLoading ? "loading…" : "selection report unavailable"} delay={160} />
          )}
        </section>
      )}

      <section className="grid" style={{ ["--min" as string]: "380px" }}>
        {!o ? (
          <>
            <CardSkeleton />
            <CardSkeleton />
          </>
        ) : (
          <>
            <Card title="Risk tiers" action={<Link to={href("/customers")} className="small">Open in explorer</Link>}
              note="Bands: low <5%, medium 5–20%, high 20–50%, critical ≥50%. They are for reading only. Who gets contacted is decided by expected value.">
              <div className="barlist">
                {o.tiers.map((t) => (
                  <Link
                    key={t.tier}
                    to={`${href("/customers")}${href("/customers").includes("?") ? "&" : "?"}tier=${t.tier}`}
                    className="barlist-row"
                    style={{ color: "var(--text)", gridTemplateColumns: "96px minmax(0,1fr) 140px" }}
                    aria-label={`${TIER_LABEL[t.tier]}: ${int(t.members)} members, ${pct(t.share)}. Open in explorer`}
                  >
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                      <span className="swatch" style={{ background: TIER_COLOR[t.tier] }} />
                      {TIER_LABEL[t.tier]}
                    </span>
                    <Track value={t.share} color={TIER_COLOR[t.tier]} />
                    <span className="num small" style={{ textAlign: "right" }}>
                      {int(t.members)} · {pct(t.share)}
                    </span>
                  </Link>
                ))}
              </div>
            </Card>

            <Card title="Revenue exposure by tier" action={<span className="num small muted">NT$ / month</span>}>
              <StackedBar
                ariaLabel={o.tiers.map((t) => `${TIER_LABEL[t.tier]} ${ntd(t.revenue_at_risk)}`).join(", ")}
                parts={[...o.tiers].reverse().map((t) => ({
                  key: t.tier,
                  value: t.revenue_at_risk,
                  color: t.tier === "low" ? "#3a3e4b" : TIER_COLOR[t.tier],
                  title: `${TIER_LABEL[t.tier]} · ${ntd(t.revenue_at_risk)}`,
                }))}
              />
              <div className="grid" style={{ ["--min" as string]: "90px", gap: 8 }}>
                {[...o.tiers].reverse().map((t) => (
                  <div key={t.tier} className="stack" style={{ ["--gap" as string]: "4px" }}>
                    <span className="small muted">{TIER_LABEL[t.tier as Tier]}</span>
                    <span className="num">{ntd(t.revenue_at_risk)}</span>
                  </div>
                ))}
              </div>
              <div className="callout">
                <span style={{ color: "var(--violet-text)", fontWeight: 500 }}>
                  {pct(highShare)} of members carry {pct(highRisk, 0)} of revenue at risk.
                </span>{" "}
                <span className="muted">The high and critical tiers are {int(highMembers)} members.</span>
              </div>
            </Card>
          </>
        )}
      </section>

      <section className="grid" style={{ ["--min" as string]: "380px" }}>
        <Card
          title="Predicted churn rate by month"
          action={
            <span className="legend">
              <span>
                <span className="swatch" style={{ border: "1.5px solid var(--violet)", borderRadius: "50%" }} />
                replayed
              </span>
              <span>
                <span className="swatch" style={{ background: "var(--violet)", borderRadius: "50%" }} />
                served
              </span>
            </span>
          }
          note="Earlier months come from the monitoring replay of the same champion model."
        >
          {history.isLoading ? (
            <CardSkeleton h={180} />
          ) : history.isError ? (
            <ErrorState error={history.error} what="History" />
          ) : trend.length < 2 ? (
            <EmptyState title="Not enough months yet">Run <code>make monitor-replay</code> to backfill earlier months.</EmptyState>
          ) : (
            <LineChart
              labels={trend.map((t) => shortMonth(t.month))}
              series={[
                {
                  name: "Mean p",
                  color: "var(--violet)",
                  values: trend.map((t) => t.mean_p),
                  hollowBefore: trend.findIndex((t) => t.served),
                },
              ]}
              format={(v) => pct(v, 2)}
              showValues
              height={210}
              ariaLabel={`Mean predicted churn by month: ${trend.map((t) => `${t.month} ${pct(t.mean_p, 2)}`).join(", ")}`}
            />
          )}
        </Card>
        <BacktestCard />
      </section>

      <section className="grid" style={{ ["--min" as string]: "380px" }}>
        {o ? (
          <Card title="Why contacted members are at risk" note={`Top reason code across the ${int(o.contacted)}-member contact list`}>
            {o.reasons.length === 0 ? (
              <EmptyState title="Nobody is on the contact list" />
            ) : (
              <div className="barlist">
                {o.reasons.slice(0, 6).map((r) => (
                  <div key={r.feature} className="stack" style={{ ["--gap" as string]: "6px" }}>
                    <div className="between" style={{ fontSize: 13 }}>
                      <span>{featureLabel(r.feature)}</span>
                      <span className="num">
                        {int(r.members)} · {pct(r.share)}
                      </span>
                    </div>
                    <Track value={r.share} color="var(--violet)" height={6} />
                  </div>
                ))}
              </div>
            )}
          </Card>
        ) : (
          <CardSkeleton />
        )}
        <HealthGlance month={month} />
      </section>
    </>
  );
}

function BacktestCard() {
  const { data, isLoading, error } = useBacktest();
  const href = useMonthHref();
  const action = <Link to={href("/retention")} className="small">Strategy details</Link>;
  if (isLoading) return <CardSkeleton />;
  if (error || !data)
    return (
      <Card title="Retention backtest" action={action}>
        <ErrorState error={error} what="Backtest" />
      </Card>
    );
  const pts = data.targeting_curve.slice(0, 2);
  const max = Math.max(...pts.flatMap((p) => [p.by_ev, p.by_risk, p.random].map(Math.abs)));
  const fmt = (v: number) => ntd(v, { signed: true }).replace("NT$", "");
  const ev = data.sensitivity.find((s) => Math.abs(s.save_rate - data.policy.save_rate) < 1e-9);
  return (
    <Card
      title="Retention backtest"
      action={action}
      note={
        <>
          NT$, assuming a {pct(data.policy.save_rate, 0)} save rate.
          {ev && (
            <>
              {" "}
              The EV rule alone picks {pct(ev.contacted_share)} of members and realises <span className="num" style={{ color: "var(--text)" }}>{ntd(ev.realised_net_value, { signed: true })}</span>.
            </>
          )}
        </>
      }
    >
      <p className="small muted">Realised net value when contacting the top slice of {int(data.members)} members</p>
      {pts.map((p) => (
        <div key={p.contact_fraction} className="stack" style={{ ["--gap" as string]: "8px" }}>
          <span className="small muted">Contact top {pct(p.contact_fraction, 0)}</span>
          <DivergingRow label="By EV" value={p.by_ev} max={max} color="var(--violet)" format={fmt} />
          <DivergingRow label="By risk" value={p.by_risk} max={max} color="var(--cyan)" format={fmt} />
          <DivergingRow label="Random" value={p.random} max={max} color="var(--slate)" format={fmt} />
        </div>
      ))}
    </Card>
  );
}

function HealthGlance({ month }: { month?: string }) {
  const { data, isLoading, error } = useMonitoring(month);
  const href = useMonthHref();
  const action = <Link to={href("/monitoring")} className="small">Monitoring</Link>;
  if (isLoading) return <CardSkeleton />;
  if (error || !data)
    return (
      <Card title="Health at a glance" action={action}>
        <ErrorState error={error} what="Monitoring report" />
      </Card>
    );
  const active = data.feature_drift.filter((f) => !f.expected_drift);
  const moderate = active.filter((f) => f.severity === "moderate").length;
  const major = active.filter((f) => f.severity === "major").length;
  const expected = data.feature_drift.filter((f) => f.expected_drift && f.severity !== "stable").length;
  const matured = data.performance.find((p) => p.status === "complete" && p.metrics);
  return (
    <Card title="Health at a glance" action={action}>
      <div className="grid" style={{ ["--min" as string]: "150px", gap: 10 }}>
        <div className="well">
          <span className="small muted">Score drift (PSI)</span>
          <span className="num" style={{ fontSize: 16 }}>{dec(data.score_drift.psi, 4)}</span>
          <span className={`small ${data.score_drift.severity === "stable" ? "pos" : "warn"}`}>{data.score_drift.severity}</span>
        </div>
        <div className="well">
          <span className="small muted">Row volume vs reference</span>
          <span className="num" style={{ fontSize: 16 }}>{dec(data.data_quality.volume_ratio, 3)}×</span>
          <span className={`small ${data.data_quality.volume_ok ? "pos" : "warn"}`}>{data.data_quality.volume_ok ? "within range" : "out of range"}</span>
        </div>
        <div className="well">
          <span className="small muted">Feature drift</span>
          <span className="num" style={{ fontSize: 16 }}>
            {major} major · {moderate} moderate
          </span>
          <span className="small muted">{expected ? `${expected} more flagged as expected` : "none expected"}</span>
        </div>
        <div className="well">
          <span className="small muted">Latest matured AUC</span>
          <span className="num" style={{ fontSize: 16 }}>{dec(matured?.metrics?.roc_auc)}</span>
          <span className="small muted">{matured ? `${shortMonth(matured.month)} · labels complete` : "no month matured yet"}</span>
        </div>
      </div>
    </Card>
  );
}
