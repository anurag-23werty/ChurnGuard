import { useState } from "react";

import { useHealth, useHistory, useMonitoring, useMonitoringMonths } from "../api/hooks";
import type { MonitoringReport } from "../api/types";
import { IconAlert, IconCheck } from "../components/icons";
import { PageHead } from "../components/Shell";
import { Card, CardSkeleton, EmptyState, ErrorState, InfoTip, Segmented } from "../components/ui";
import { dec, int, monthLabel, pct, shortMonth } from "../lib/format";
import { useMonth } from "../lib/month";

const PSI_AXIS = 0.3;
const MODERATE = 0.1;
const MAJOR = 0.25;

export function MonitoringPage() {
  const { month: scoreMonth } = useMonth();
  const months = useMonitoringMonths();
  const available = months.data?.months ?? [];
  const [picked, setPicked] = useState<string | undefined>();
  const month = picked && available.includes(picked) ? picked : scoreMonth && available.includes(scoreMonth) ? scoreMonth : available[0];
  const report = useMonitoring(month);

  return (
    <>
      <PageHead
        title="Monitoring"
        month={false}
        actions={
          available.length > 1 ? (
            <Segmented label="Report month" value={month ?? ""} onChange={setPicked} options={available.map((m) => ({ value: m, label: shortMonth(m) }))} />
          ) : undefined
        }
      >
        {report.data ? `${monthLabel(report.data.month)} compared with ${report.data.reference_months[0]} to ${report.data.reference_months.at(-1)}` : "Data quality, drift and delayed-label performance"}
      </PageHead>

      <ApiCard />

      {months.isLoading || report.isLoading ? (
        <>
          <CardSkeleton h={60} />
          <CardSkeleton h={240} />
        </>
      ) : months.isError ? (
        <Card>
          <ErrorState error={months.error} what="Monitoring reports" />
        </Card>
      ) : !month ? (
        <Card>
          <EmptyState title="No monitoring report yet">
            Run <code>make monitor</code> after scoring a month.
          </EmptyState>
        </Card>
      ) : report.isError || !report.data ? (
        <Card>
          <ErrorState error={report.error} what="Monitoring report" retry={() => report.refetch()} />
        </Card>
      ) : (
        <Report r={report.data} />
      )}
      <ReplayCard />
    </>
  );
}

function ApiCard() {
  const { data, error, isLoading, dataUpdatedAt } = useHealth();
  const ok = data?.status === "ok";
  return (
    <section className="grid" style={{ ["--min" as string]: "300px" }}>
      <Card title="API" action={<code className="inline">GET /health</code>}>
        {isLoading ? (
          <span className="muted small">Checking…</span>
        ) : (
          <div>
            <div className="kv">
              <span>Status</span>
              <span className={ok ? "pos" : "warn"}>{ok ? "healthy" : error ? "unreachable" : data?.status}</span>
            </div>
            <div className="kv">
              <span>Serving model</span>
              <span className="num">{data ? `${data.model.name} v${data.model.version} · ${data.model.model_type}` : "—"}</span>
            </div>
            <div className="kv">
              <span>Registry alias</span>
              <span className="num">{data?.model.alias ?? "—"}</span>
            </div>
            <div className="kv">
              <span>Last checked</span>
              <span className="num">{dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : "—"}</span>
            </div>
          </div>
        )}
      </Card>
      <Card title="Latency and error rate">
        <div className="dashed">
          <strong style={{ color: "var(--text)", fontWeight: 500 }}>Unavailable from the current API</strong>
          <div>The service exposes no request-metrics endpoint, so latency percentiles and error rates cannot be charted. Health is polled every 30 seconds.</div>
        </div>
      </Card>
    </section>
  );
}

function Report({ r }: { r: MonitoringReport }) {
  const [hideExpected, setHideExpected] = useState(false);
  const d = r.decision;
  const tone = d.status === "ok" ? "ok" : "bad";
  const feats = r.feature_drift.filter((f) => !(hideExpected && f.expected_drift)).slice(0, 12);

  return (
    <>
      <section
        role="status"
        className={`pill ${tone}`}
        style={{ height: "auto", padding: "16px 20px", borderRadius: 12, display: "flex", flexWrap: "wrap", gap: 16, alignItems: "center", color: "var(--text)" }}
      >
        <span style={{ width: 36, height: 36, borderRadius: 10, display: "grid", placeItems: "center", background: d.status === "ok" ? "#123a3e" : "#3a2c10" }}>
          {d.status === "ok" ? <IconCheck color="var(--cyan)" /> : <IconAlert color="var(--amber)" />}
        </span>
        <span className="stack" style={{ flex: "1 1 280px", ["--gap" as string]: "2px" }}>
          <strong style={{ fontSize: 15 }}>
            {d.status === "ok" ? "OK · no retraining needed" : d.status === "warn" ? "Warning · review before the next campaign" : "Retrain recommended"}
          </strong>
          <span className="muted" style={{ fontSize: 13 }}>
            {[...d.retrain_reasons, ...d.warnings].join("; ") || "No retrain reasons and no warnings."}
          </span>
        </span>
        <code className="inline">make monitor MONTH={r.month}</code>
      </section>

      <section className="grid" style={{ ["--min" as string]: "300px" }}>
        <Card title="Data quality">
          <div>
            <div className="kv"><span>Rows scored</span><span className="num">{int(r.data_quality.rows)}</span></div>
            <div className="kv">
              <span>Volume vs reference</span>
              <span className="num">
                {dec(r.data_quality.volume_ratio, 3)}× <span className={r.data_quality.volume_ok ? "pos" : "warn"}>{r.data_quality.volume_ok ? "ok" : "out of range"}</span>
              </span>
            </div>
            <div className="kv">
              <span>Largest null-rate jump</span>
              <span className="num">{r.data_quality.max_null_jump ? `${r.data_quality.max_null_jump.feature} ${r.data_quality.max_null_jump.delta >= 0 ? "+" : "−"}${Math.abs(r.data_quality.max_null_jump.delta * 100).toFixed(1)} pt` : "none"}</span>
            </div>
            <div className="kv"><span>Probabilities in [0, 1]</span><span className={`num ${r.data_quality.p_in_range ? "pos" : "warn"}`}>{r.data_quality.p_in_range ? "yes" : "no"}</span></div>
          </div>
        </Card>
        <Card
          title="Prediction drift"
          note="Input drift is a pipeline health check. Retraining is triggered by matured performance: December 2016 drifted in outcome while its inputs stayed stable."
        >
          <div>
            <div className="kv"><span>Score PSI</span><span className="num">{dec(r.score_drift.psi, 4)} <span className={r.score_drift.severity === "stable" ? "pos" : "warn"}>{r.score_drift.severity}</span></span></div>
            <div className="kv"><span>Mean p, reference</span><span className="num">{pct(r.score_drift.mean_p_reference, 2)}</span></div>
            <div className="kv"><span>Mean p, {shortMonth(r.month)}</span><span className="num">{pct(r.score_drift.mean_p_current, 2)}</span></div>
          </div>
        </Card>
      </section>

      <Card
        title={`Feature drift · PSI, top ${feats.length} of ${r.feature_drift.length}`}
        action={
          <span style={{ display: "inline-flex", gap: 12, alignItems: "center" }}>
            <InfoTip label="About PSI">Population stability index between the reference months and this month. Below 0.10 is stable, 0.10–0.25 moderate, above 0.25 major.</InfoTip>
            <label className="check">
              <input type="checkbox" checked={hideExpected} onChange={(e) => setHideExpected(e.target.checked)} />
              Hide expected drift
            </label>
          </span>
        }
        note={`The axis stops at ${PSI_AXIS}. Features flagged expected_drift in the registry (clock-like features) never raise alerts.`}
      >
        <div className="barlist" style={{ gap: 10 }}>
          {feats.map((f) => {
            const color = f.expected_drift ? "var(--slate)" : f.severity === "major" ? "var(--amber)" : f.severity === "moderate" ? "var(--violet)" : "var(--cyan)";
            return (
              <div key={f.feature} className="barlist-row" style={{ gridTemplateColumns: "minmax(0, 210px) minmax(0, 1fr) 150px" }}>
                <span className="num small" style={{ overflow: "hidden", textOverflow: "ellipsis" }} title={f.feature}>
                  {f.feature}
                </span>
                <div style={{ position: "relative" }}>
                  <div className="track" style={{ height: 10 }}>
                    <span style={{ width: `${Math.min(100, (f.psi / PSI_AXIS) * 100)}%`, background: color }} />
                  </div>
                  <span aria-hidden="true" style={{ position: "absolute", left: `${(MODERATE / PSI_AXIS) * 100}%`, top: -3, width: 1, height: 16, background: "var(--slate)" }} />
                  <span aria-hidden="true" style={{ position: "absolute", left: `${(MAJOR / PSI_AXIS) * 100}%`, top: -3, width: 1, height: 16, background: "#a87a22" }} />
                </div>
                <span className={`num small ${f.severity === "stable" ? "muted" : ""}`} style={{ textAlign: "right" }}>
                  {f.psi.toFixed(3)} · {f.expected_drift ? "expected" : f.severity}
                </span>
              </div>
            );
          })}
        </div>
      </Card>

      <Card flush title="Performance as labels mature" action={<span className="small muted">churn is known 30 days after expiry, so recent months fill in over time</span>}>
        <div className="table-wrap">
          <table className="data" style={{ minWidth: 620 }}>
            <thead>
              <tr>
                <th scope="col">Month</th>
                <th scope="col">Labels</th>
                <th scope="col" className="r">Decided</th>
                <th scope="col" className="r">ROC AUC</th>
                <th scope="col" className="r">PR AUC</th>
                <th scope="col">Counts toward retraining?</th>
              </tr>
            </thead>
            <tbody>
              {r.performance.map((p) => (
                <tr key={p.month}>
                  <td className="num">{p.month}</td>
                  <td>
                    <span className={`badge ${p.status === "complete" ? "ok" : "warn"}`}>{p.status}</span>
                  </td>
                  <td className="r num">
                    {int(p.decided)} · {pct(p.decided_share)}
                  </td>
                  <td className={`r num ${p.status !== "complete" ? "faint" : ""}`}>{dec(p.metrics?.roc_auc)}</td>
                  <td className={`r num ${p.status !== "complete" ? "faint" : ""}`}>{dec(p.metrics?.pr_auc)}</td>
                  <td className="small muted">
                    {p.in_sample ? "no: month is in the training data" : p.status !== "complete" ? "no: labels still maturing" : "yes"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}

function ReplayCard() {
  const { data, isLoading, error } = useHistory();
  if (isLoading) return <CardSkeleton />;
  const replay = data?.replay;
  return (
    <Card flush title="Drift replay" action={<span className="small muted">each month against the two before it</span>}>
      {error ? (
        <ErrorState error={error} what="Replay" />
      ) : !replay ? (
        <EmptyState title="No replay yet">Run <code>make monitor-replay</code> to backfill drift history.</EmptyState>
      ) : (
        <>
          <div className="table-wrap">
            <table className="data" style={{ minWidth: 620 }}>
              <thead>
                <tr>
                  <th scope="col">Month</th>
                  <th scope="col" className="r">Score PSI</th>
                  <th scope="col" className="r">Mean p</th>
                  <th scope="col" className="r">Moderate / major</th>
                  <th scope="col">Largest drift</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <tbody>
                {[...replay].reverse().map((r) => (
                  <tr key={r.month}>
                    <td className="num">{r.month}</td>
                    <td className="r num">{r.score_psi.toFixed(4)}</td>
                    <td className="r num">{pct(r.mean_p, 2)}</td>
                    <td className="r num">
                      {r.moderate} / {r.major}
                    </td>
                    <td className="num small muted">{r.top_drift.split(",")[0]}</td>
                    <td>
                      <span className={`badge ${r.decision === "ok" ? "ok" : "warn"}`}>{r.decision}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="card-note warn">
            Input drift stayed quiet in December 2016, yet the model's AUC fell that month. Only the delayed-label rule catches outcome drift like this, about 30 days after the month closes.
          </p>
        </>
      )}
    </Card>
  );
}
