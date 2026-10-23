import { useState } from "react";

import { useBaselines, useSelection } from "../api/hooks";
import type { Metrics, Selection } from "../api/types";
import { LineChart } from "../components/charts";
import { PageHead } from "../components/Shell";
import { Card, CardSkeleton, ErrorState, InfoTip, Segmented, Track } from "../components/ui";
import { CANDIDATE_LABEL, dec, int, pct, shortMonth } from "../lib/format";

type MetricKey = "roc_auc" | "pr_auc" | "log_loss" | "ece" | "lift_top10";
const METRICS: { key: MetricKey; label: string; note: string; fmt: (v: number) => string }[] = [
  { key: "roc_auc", label: "ROC AUC", note: "Higher is better.", fmt: (v) => v.toFixed(3) },
  { key: "pr_auc", label: "PR AUC", note: "Higher is better; the base rate is around 5–9%.", fmt: (v) => v.toFixed(3) },
  { key: "log_loss", label: "Log loss", note: "Lower is better.", fmt: (v) => v.toFixed(3) },
  { key: "ece", label: "ECE", note: "Calibration error: lower is better.", fmt: (v) => v.toFixed(4) },
  { key: "lift_top10", label: "Lift @10%", note: "How many times the base rate the top 10% catches.", fmt: (v) => `${v.toFixed(1)}×` },
];
const COLORS: Record<string, { color: string; dashed?: boolean }> = {
  logreg: { color: "var(--violet)" },
  lgbm: { color: "var(--cyan)" },
  lgbm_robust: { color: "#1f8f96", dashed: true },
  blend_lgbm_logreg: { color: "var(--amber)" },
  blend_robust_logreg: { color: "#a87a22", dashed: true },
};
const label = (c: string) => CANDIDATE_LABEL[c] ?? c;

export function PerformancePage() {
  const selection = useSelection();
  const baselines = useBaselines();
  return (
    <>
      <PageHead title="Model performance" month={false}>
        Every month is scored by a model trained only on earlier months. The holdout month is scored against KKBox's official labels.
      </PageHead>
      {selection.isLoading ? (
        <>
          <CardSkeleton h={260} />
          <CardSkeleton h={200} />
        </>
      ) : selection.isError || !selection.data ? (
        <Card>
          <ErrorState error={selection.error} what="Model selection report" retry={() => selection.refetch()} />
        </Card>
      ) : (
        <SelectionView s={selection.data} />
      )}
      <section className="grid" style={{ ["--min" as string]: "320px" }}>
        {selection.data && <CalibrationCard s={selection.data} />}
        {baselines.isLoading ? (
          <CardSkeleton />
        ) : baselines.isError || !baselines.data ? (
          <Card title="Against the baselines">
            <ErrorState error={baselines.error} what="Baselines report" />
          </Card>
        ) : (
          <Card
            title="Against the baselines · validation month"
            note={(() => {
              const pc = baselines.data._permutation_control;
              const se = pc.sd / Math.sqrt(pc.k);
              return `ROC AUC. Shuffled-label control: ${pc.k} permutations averaged ${pc.mean.toFixed(3)} ± ${se.toFixed(3)} (SE), close to 0.5, so no label leakage.`;
            })()}
          >
            <div className="barlist">
              {(
                [
                  ["prior", "Prior (same p for everyone)", "var(--slate)"],
                  ["rule", "Rule: auto-renew × cancel", "var(--cyan)"],
                  ["logreg", "Logistic regression", "var(--violet)"],
                ] as const
              ).map(([k, name, color]) => (
                <div key={k} className="stack" style={{ ["--gap" as string]: "6px" }}>
                  <div className="between" style={{ fontSize: 13 }}>
                    <span>{name}</span>
                    <span className="num">{dec(baselines.data[k].valid.roc_auc)}</span>
                  </div>
                  <Track value={baselines.data[k].valid.roc_auc} color={color} height={6} />
                </div>
              ))}
            </div>
          </Card>
        )}
        {selection.data && <ConfusionCard s={selection.data} />}
      </section>
    </>
  );
}

function SelectionView({ s }: { s: Selection }) {
  const [metric, setMetric] = useState<MetricKey>("roc_auc");
  const [shown, setShown] = useState<Set<string>>(() => new Set([s.selected, "lgbm", "blend_lgbm_logreg"].filter((c) => s.candidates.includes(c))));
  const [pop, setPop] = useState<string>("excluding_early_renewers");
  const m = METRICS.find((x) => x.key === metric)!;
  const months = s.scored_months;
  const champRows = s.per_month.filter((r) => r.candidate === s.selected);
  const worst = champRows.reduce((w, r) => (r.roc_auc < w.roc_auc ? r : w), champRows[0]);
  const series = s.candidates
    .filter((c) => shown.has(c))
    .map((c) => ({
      name: label(c),
      color: COLORS[c]?.color ?? "var(--text-muted)",
      dashed: COLORS[c]?.dashed,
      width: c === s.selected ? 3 : 2,
      values: months.map((mo) => {
        const r = s.per_month.find((x) => x.candidate === c && x.month === mo);
        return r ? (r[metric] as number) : null;
      }),
    }));
  const test = s.test[s.selected] ?? {};
  const pops = Object.keys(test);
  const t: Metrics | undefined = test[pop] ?? test[pops[0]];

  return (
    <>
      <Card
        title="Out-of-time results by month"
        action={<Segmented label="Metric" value={metric} onChange={setMetric} options={METRICS.map((x) => ({ value: x.key, label: x.label }))} />}
        note={`${m.note} The shaded month is the champion's worst; promotion is judged on it as well as on the mean.`}
      >
        <div role="group" aria-label="Candidates shown" style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {s.candidates.map((c) => {
            const on = shown.has(c);
            return (
              <button
                key={c}
                type="button"
                className="btn"
                aria-pressed={on}
                style={{ minHeight: 34, fontSize: 12, color: on ? "var(--text)" : "var(--text-faint)" }}
                onClick={() =>
                  setShown((prev) => {
                    const n = new Set(prev);
                    if (n.has(c)) n.delete(c);
                    else n.add(c);
                    return n;
                  })
                }
              >
                <span style={{ width: 16, borderTop: `2px ${COLORS[c]?.dashed ? "dashed" : "solid"} ${COLORS[c]?.color ?? "var(--text-muted)"}`, opacity: on ? 1 : 0.35 }} />
                {label(c)}
              </button>
            );
          })}
        </div>
        {series.length === 0 ? (
          <p className="muted small">Select at least one candidate.</p>
        ) : (
          <LineChart
            labels={months.map(shortMonth)}
            series={series}
            format={m.fmt}
            height={260}
            band={worst ? { index: months.indexOf(worst.month), label: "worst month" } : undefined}
            ariaLabel={`${m.label} by month for ${series.map((x) => x.name).join(", ")}`}
          />
        )}
      </Card>

      <Card flush title="Promotion gate v2" action={<span className="small muted">beat the baseline on average, and stay within 0.01 AUC of it in the worst month</span>}>
        <div className="table-wrap">
          <table className="data" style={{ minWidth: 860 }}>
            <thead>
              <tr>
                <th scope="col">Candidate</th>
                <th scope="col" className="r">Mean AUC</th>
                <th scope="col" className="r">Worst AUC</th>
                <th scope="col" className="r">Mean PR AUC</th>
                <th scope="col" className="r">Mean log loss</th>
                <th scope="col">Gate</th>
              </tr>
            </thead>
            <tbody>
              {[...s.summary]
                .sort((a, b) => (a.candidate === s.selected ? -1 : b.candidate === s.selected ? 1 : b.mean_roc_auc - a.mean_roc_auc))
                .map((r) => {
                  const g = s.gate[r.candidate];
                  const champ = r.candidate === s.selected;
                  const reasons = g?.reasons.join("; ") ?? "";
                  return (
                    <tr key={r.candidate} className={champ ? "highlight" : ""}>
                      <td>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                          <span className="swatch" style={{ borderRadius: "50%", background: COLORS[r.candidate]?.color }} />
                          {label(r.candidate)}
                        </span>
                      </td>
                      <td className="r num">{dec(r.mean_roc_auc, 4)}</td>
                      <td className={`r num ${!champ && /worst-month/.test(reasons) ? "warn" : ""}`}>{dec(r.worst_roc_auc, 4)}</td>
                      <td className="r num">{dec(r.mean_pr_auc, 4)}</td>
                      <td className={`r num ${!champ && /log loss/.test(reasons) ? "warn" : ""}`}>{dec(r.mean_log_loss, 4)}</td>
                      <td style={{ whiteSpace: "normal", minWidth: 260 }}>
                        {champ ? (
                          <span className="badge violet">Selected · baseline</span>
                        ) : g?.promote ? (
                          <span className="badge ok">Passes</span>
                        ) : (
                          <span className="small muted">Blocked: {reasons || "did not beat the selected model"}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </Card>

      {t && (
        <Card
          title={`Holdout test · official labels`}
          action={
            pops.length > 1 ? (
              <Segmented
                label="Test population"
                value={pop in test ? pop : pops[0]}
                onChange={setPop}
                options={pops.map((p) => ({ value: p, label: p === "all" ? "All members" : "Excl. early renewers" }))}
              />
            ) : undefined
          }
          note={
            pop === "excluding_early_renewers"
              ? "Early renewers are removed: they had already paid beyond the test month, so it cannot observe their decision."
              : undefined
          }
        >
          <p className="small muted">
            {int(t.n)} members · {pct(t.base_rate)} churned
          </p>
          <div className="grid" style={{ ["--min" as string]: "140px", gap: 10 }}>
            <div className="well"><span className="small muted">ROC AUC</span><span className="num" style={{ fontSize: 20 }}>{dec(t.roc_auc)}</span></div>
            <div className="well"><span className="small muted">PR AUC</span><span className="num" style={{ fontSize: 20 }}>{dec(t.pr_auc)}</span></div>
            <div className="well"><span className="small muted">Precision @ top 10%</span><span className="num" style={{ fontSize: 20 }}>{pct(t.precision_top10)}</span></div>
            <div className="well"><span className="small muted">Recall @ top 10%</span><span className="num" style={{ fontSize: 20 }}>{pct(t.recall_top10)}</span></div>
            <div className="well"><span className="small muted">Lift @ top 10%</span><span className="num" style={{ fontSize: 20 }}>{t.lift_top10 ? `${t.lift_top10.toFixed(2)}×` : "—"}</span></div>
            <div className="well"><span className="small muted">Calibration error</span><span className="num" style={{ fontSize: 20 }}>{dec(t.ece)}</span></div>
          </div>
        </Card>
      )}
    </>
  );
}

function CalibrationCard({ s }: { s: Selection }) {
  const rows = s.per_month.filter((r) => r.candidate === s.selected);
  const max = Math.max(...rows.map((r) => r.ece), 1e-6);
  const c = s.calibration;
  return (
    <Card
      title="Calibration · selected model"
      action={<InfoTip label="About ECE">Expected calibration error: the average gap between predicted probability and observed churn rate, weighted by bin size.</InfoTip>}
      note={
        c.applied
          ? `Isotonic recalibration is applied: it improved ${c.holdout_month} (${c.raw_ece.toFixed(4)} → ${c.cal_ece.toFixed(4)}).`
          : `Isotonic recalibration was tested on ${c.holdout_month} and made it worse (${c.raw_ece.toFixed(4)} → ${c.cal_ece.toFixed(4)}), so it is not applied.`
      }
    >
      <span className="small muted">Expected calibration error per out-of-time month</span>
      <div className="barlist" style={{ gap: 10 }}>
        {rows.map((r) => (
          <div key={r.month} className="barlist-row" style={{ gridTemplateColumns: "56px minmax(0,1fr) 60px", fontSize: 12 }}>
            <span className="num muted">{shortMonth(r.month)}</span>
            <Track value={r.ece / max} color={r.ece === max ? "var(--amber)" : "var(--violet)"} />
            <span className="num" style={{ textAlign: "right" }}>{r.ece.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

/** Confusion matrix at the top-10% operating point, derived from reported precision and
 *  recall. Decisions use expected value, not a threshold, so this is one reference point. */
function ConfusionCard({ s }: { s: Selection }) {
  const t = s.test[s.selected]?.all;
  if (!t?.n || t.base_rate === undefined || t.precision_top10 === undefined) {
    return (
      <Card title="Confusion matrix">
        <p className="muted small">The test report has no precision at a fixed cut-off, so a confusion matrix cannot be derived.</p>
      </Card>
    );
  }
  const flagged = Math.round(t.n * 0.1);
  const pos = Math.round(t.n * t.base_rate);
  const tp = Math.round(flagged * t.precision_top10);
  const fp = flagged - tp;
  const fn = pos - tp;
  const tn = t.n - flagged - fn;
  const cell = (v: number, title: string, strong: boolean) => (
    <div className="well" style={{ alignItems: "center", textAlign: "center", background: strong ? "var(--violet-bg-2)" : undefined, borderColor: strong ? "var(--violet-line)" : undefined }}>
      <span className="small muted">{title}</span>
      <span className="num" style={{ fontSize: 18 }}>{int(v)}</span>
    </div>
  );
  return (
    <Card
      title="Confusion matrix · top 10% flagged"
      note="Derived from the holdout's reported precision and recall at the top-10% cut-off (all members). The live decision uses expected value, not a fixed threshold."
    >
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 8, alignItems: "center", fontSize: 12 }}>
        <span />
        <span className="muted" style={{ textAlign: "center" }}>Churned</span>
        <span className="muted" style={{ textAlign: "center" }}>Renewed</span>
        <span className="muted">Flagged</span>
        {cell(tp, "true positive", true)}
        {cell(fp, "false positive", false)}
        <span className="muted">Not flagged</span>
        {cell(fn, "false negative", false)}
        {cell(tn, "true negative", false)}
      </div>
    </Card>
  );
}
