import { useEffect, useState } from "react";

import { useBacktest, usePlan, usePolicy } from "../api/hooks";
import { LineChart } from "../components/charts";
import { ExportButton } from "../components/ExportButton";
import { PageHead } from "../components/Shell";
import { Card, CardSkeleton, ErrorState, InfoTip, Segmented, Skeleton } from "../components/ui";
import { int, monthLabel, ntd, pct } from "../lib/format";
import { useMonth } from "../lib/month";

const PRESETS = [null, 100_000, 500_000, 1_000_000] as const;

export function RetentionPage() {
  const { month } = useMonth();
  const backtest = useBacktest();
  const policy = usePolicy();

  return (
    <>
      <PageHead title="Retention intelligence">
        Contact a member only when p × save rate × CLV is worth more than the offer
      </PageHead>
      <BudgetPlanner month={month} />
      {backtest.isLoading ? (
        <CardSkeleton />
      ) : backtest.isError || !backtest.data ? (
        <Card title="Backtest">
          <ErrorState error={backtest.error} what="Retention backtest" />
        </Card>
      ) : (
        <Backtest data={backtest.data} />
      )}
      {policy.data && (
        <Card title="Policy in force" note={<>Set in <code className="inline">configs/base.yaml</code> under <code className="inline">retention</code>. With a budget, members are picked greedily by EV per unit of cost.</>}>
          <div className="grid" style={{ ["--min" as string]: "180px", gap: 0, columnGap: 24 }}>
            <div className="kv"><span>Save rate</span><span className="num">{pct(policy.data.save_rate, 0)} · assumed</span></div>
            <div className="kv"><span>CLV horizon</span><span className="num">{policy.data.horizon_months} months</span></div>
            <div className="kv"><span>Baseline monthly churn</span><span className="num">{pct(policy.data.baseline_monthly_churn, 0)}</span></div>
            <div className="kv"><span>Contact cost</span><span className="num">NT${policy.data.contact_cost}</span></div>
            <div className="kv"><span>Incentive</span><span className="num">{policy.data.incentive_months} free month{policy.data.incentive_months === 1 ? "" : "s"}</span></div>
            <div className="kv"><span>Default budget</span><span className="num">{policy.data.budget === null ? "none · EV > 0" : ntd(policy.data.budget)}</span></div>
          </div>
        </Card>
      )}
    </>
  );
}

function BudgetPlanner({ month }: { month?: string }) {
  const [draft, setDraft] = useState("");
  const [budget, setBudget] = useState<number | null>(null);
  const invalid = draft.trim() !== "" && !(Number(draft) >= 0);
  useEffect(() => {
    if (invalid) return;
    const t = window.setTimeout(() => setBudget(draft.trim() === "" ? null : Math.round(Number(draft))), 400);
    return () => window.clearTimeout(t);
  }, [draft, invalid]);
  const plan = usePlan(month, budget);
  const d = plan.data;

  return (
    <Card
      title={`Campaign planner${month ? ` · ${monthLabel(month)}` : ""}`}
      action={<ExportButton month={month} contacted={d?.contacted} budget={budget} label="Export this list" />}
    >
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
        <label className="field" style={{ flex: "0 1 220px" }}>
          Campaign budget (NT$)
          <input
            className="input num"
            inputMode="numeric"
            placeholder="No cap"
            value={draft}
            onChange={(e) => setDraft(e.target.value.replace(/[, ]/g, ""))}
            aria-invalid={invalid}
          />
        </label>
        <Segmented
          label="Budget presets"
          value={budget === null ? "none" : String(budget)}
          onChange={(v) => setDraft(v === "none" ? "" : v)}
          options={PRESETS.map((p) => ({ value: p === null ? "none" : String(p), label: p === null ? "No cap" : ntd(p) }))}
        />
        {invalid && <span className="small warn" role="alert">Enter a non-negative number.</span>}
      </div>
      {plan.isError ? (
        <ErrorState error={plan.error} what="Campaign plan" retry={() => plan.refetch()} />
      ) : !d ? (
        <Skeleton h={88} />
      ) : (
        <div className="grid" style={{ ["--min" as string]: "170px", opacity: plan.isPlaceholderData ? 0.6 : 1, transition: "opacity .15s" }} aria-live="polite">
          <div className="well">
            <span className="small muted">Members to contact</span>
            <span className="num" style={{ fontSize: 22 }}>{int(d.contacted)}</span>
            <span className="small muted">{pct(d.contact_share, 2)} of {int(d.members)}</span>
          </div>
          <div className="well">
            <span className="small muted">Campaign cost</span>
            <span className="num" style={{ fontSize: 22 }}>{ntd(d.campaign_cost)}</span>
            <span className="small muted">{d.budget === null ? "no cap" : `of ${ntd(d.budget)} budget`}</span>
          </div>
          <div className="well">
            <span className="small muted">Expected benefit</span>
            <span className="num" style={{ fontSize: 22 }}>{ntd(d.expected_benefit)}</span>
            <span className="small muted">churners saved × CLV</span>
          </div>
          <div className="well" style={{ background: "var(--violet-bg-2)", borderColor: "var(--violet-line)" }}>
            <span className="small" style={{ color: "var(--violet-text)", display: "flex", justifyContent: "space-between" }}>
              Expected net value
              <InfoTip label="About expected net value">Model-based expectation for this month, not a measured result. It assumes the {pct(d.policy.save_rate, 0)} save rate.</InfoTip>
            </span>
            <span className="num pos" style={{ fontSize: 22 }}>{ntd(d.expected_net_value, { signed: true })}</span>
            <span className="small muted">lowest risk contacted {pct(d.min_p_contacted)}</span>
          </div>
        </div>
      )}
    </Card>
  );
}

function Backtest({ data }: { data: NonNullable<ReturnType<typeof useBacktest>["data"]> }) {
  const [rate, setRate] = useState(String(data.policy.save_rate));
  const cur = data.sensitivity.find((s) => String(s.save_rate) === rate) ?? data.sensitivity[0];
  const curve = data.targeting_curve;
  const best = data.best_fixed_fraction_by_ev;

  return (
    <>
      <Card
        title={`Save-rate sensitivity · backtest on ${int(data.members)} members, ${pct(data.churn_rate)} churned`}
        action={
          <Segmented
            label="Assumed save rate"
            value={rate}
            onChange={setRate}
            options={data.sensitivity.map((s) => ({ value: String(s.save_rate), label: <span className="num">{pct(s.save_rate, 0)}</span> }))}
          />
        }
      >
        <p className="small muted">The share of contacted churners an offer keeps. KKBox has no campaign data, so this input is assumed. Results are realised against the labels that actually happened.</p>
        <div className="grid" style={{ ["--min" as string]: "180px" }}>
          <div className="well">
            <span className="small muted">Members contacted</span>
            <span className="num" style={{ fontSize: 22 }}>{int(cur.contacted)}</span>
            <span className="small muted num">{pct(cur.contacted_share, 2)} of members</span>
          </div>
          <div className="well">
            <span className="small muted">Break-even risk (NT$149 plan)</span>
            <span className="num" style={{ fontSize: 22 }}>{cur.break_even_p_at_149 > 1 ? ">100%" : pct(cur.break_even_p_at_149)}</span>
            <span className="small muted">{cur.break_even_p_at_149 > 1 ? "no member can qualify" : "p needed before an offer pays"}</span>
          </div>
          <div className="well" style={{ background: "var(--violet-bg-2)", borderColor: "var(--violet-line)" }}>
            <span className="small" style={{ color: "var(--violet-text)" }}>Realised net value · EV rule</span>
            <span className="num" style={{ fontSize: 22 }}>{ntd(cur.realised_net_value, { signed: true })}</span>
            <span className="small muted">saved churners × CLV, minus offer cost</span>
          </div>
          <div className="well">
            <span className="small muted">If you contacted everyone</span>
            <span className="num muted" style={{ fontSize: 22 }}>{ntd(cur.realised_value_contact_everyone, { signed: true })}</span>
            <span className="small muted">cost of a blanket offer</span>
          </div>
        </div>
        {cur.contacted === 0 && (
          <div className="dashed" role="status">
            At a {pct(cur.save_rate, 0)} save rate no member's expected benefit covers the offer, so the EV rule contacts nobody. That is the correct answer, not an error.
          </div>
        )}
      </Card>

      <section className="row">
        <Card
          className="grow"
          title="Targeting strategies · net value by campaign size"
          action={
            <span className="legend">
              <span><span className="swatch" style={{ background: "var(--violet)" }} />By EV</span>
              <span><span className="swatch" style={{ background: "var(--cyan)" }} />By risk</span>
              <span><span className="swatch" style={{ background: "var(--slate)" }} />Random</span>
            </span>
          }
          note={`Share of members contacted (points evenly spaced). At a ${pct(data.policy.save_rate, 0)} save rate the best fixed cut-off is the top ${pct(best.contact_fraction, 0)} (${ntd(best.realised_net_value, { signed: true })}).`}
        >
          <LineChart
            labels={curve.map((c) => pct(c.contact_fraction, 0))}
            series={[
              { name: "Random", color: "var(--slate)", values: curve.map((c) => c.random), dashed: true },
              { name: "By risk", color: "var(--cyan)", values: curve.map((c) => c.by_risk) },
              { name: "By EV", color: "var(--violet)", values: curve.map((c) => c.by_ev), width: 2.5 },
            ]}
            format={(v) => ntd(v).replace("NT$", "")}
            height={240}
            ariaLabel={`Net value by share contacted. By EV: ${curve.map((c) => `${pct(c.contact_fraction, 0)} ${ntd(c.by_ev)}`).join(", ")}`}
          />
        </Card>
        <Card className="side" title="What this tells a campaign owner">
          <ol className="stack" style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.5, ["--gap" as string]: "10px" }}>
            <li>The EV rule sizes the campaign by itself and lands near the best cut-off found in hindsight, with no tuning.</li>
            <li>That works only because the probabilities are calibrated. A miscalibrated model would size the list wrongly.</li>
            <li>The ranking (EV ahead of risk ahead of random) holds at every save rate tested. The NT$ amounts do not, so measure the save rate with a holdout group first.</li>
          </ol>
          <div className="dashed">Uplift estimates are unavailable: there are no treatment and control records to train an uplift model on.</div>
        </Card>
      </section>

      <Card flush title="Strategy comparison · realised NT$">
        <div className="table-wrap">
          <table className="data" style={{ minWidth: 560 }}>
            <thead>
              <tr>
                <th scope="col">Contact</th>
                <th scope="col" className="r">Members</th>
                <th scope="col" className="r">By EV</th>
                <th scope="col" className="r">By risk</th>
                <th scope="col" className="r">Random</th>
              </tr>
            </thead>
            <tbody>
              {curve.map((c) => (
                <tr key={c.contact_fraction} className={c.contact_fraction === best.contact_fraction ? "highlight" : ""}>
                  <td className="num">{pct(c.contact_fraction, 0)}</td>
                  <td className="r num">{int(c.contacted)}</td>
                  {[c.by_ev, c.by_risk, c.random].map((v, i) => (
                    <td key={i} className={`r num ${v > 0 ? "" : "muted"}`} style={v > 0 && i === 0 ? { color: "var(--violet-text)" } : undefined}>
                      {v > 0 ? "+" : v < 0 ? "−" : ""}
                      {int(Math.abs(v))}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
