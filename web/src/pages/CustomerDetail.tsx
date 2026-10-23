import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api, ApiError } from "../api/client";
import { useMember } from "../api/hooks";
import type { MemberDetail } from "../api/types";
import { Gauge } from "../components/charts";
import { IconCopy } from "../components/icons";
import { PageHead, useMonthHref } from "../components/Shell";
import { useToast } from "../components/toast";
import { Card, CardSkeleton, ErrorState, InfoTip, TierBadge, Track } from "../components/ui";
import { FEATURES, featureLabel, type Family } from "../lib/features";
import { dateTime, hours, int, isoDate, monthStart, ntd, pct, TIER_COLOR } from "../lib/format";
import { useMonth } from "../lib/month";

type Value = string | number | null | undefined;

function formatFeature(name: string, v: Value): string {
  if (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v))) return "—";
  if (typeof v === "string") return v;
  switch (FEATURES[name]?.unit) {
    case "days":
      return `${int(v)} d`;
    case "secs":
      return hours(v);
    case "ntd":
      return `NT$${int(v)}`;
    case "share":
      return pct(v);
    case "flag":
      return v ? "yes" : "no";
    case "code":
      return String(v);
    default:
      return Number.isInteger(v) ? int(v) : v.toFixed(2);
  }
}

export function CustomerDetailPage() {
  const { id } = useParams();
  const memberId = Number(id);
  const { month } = useMonth();
  const href = useMonthHref();
  const { data, error, isLoading, refetch } = useMember(memberId, month);

  const crumbs = (
    <nav aria-label="Breadcrumb" className="small muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
      <Link to={href("/customers")} className="muted" style={{ minHeight: 32, display: "inline-flex", alignItems: "center" }}>
        Customers
      </Link>
      <span aria-hidden="true">/</span>
      <span className="num" aria-current="page" style={{ color: "var(--text)" }}>
        {Number.isFinite(memberId) ? memberId : id}
      </span>
    </nav>
  );

  if (!Number.isFinite(memberId)) {
    return (
      <>
        {crumbs}
        <Card>
          <ErrorState error={new ApiError(422, `"${id}" is not a member ID`)} what="Member" />
        </Card>
      </>
    );
  }

  return (
    <>
      {crumbs}
      {isLoading || (!data && !error) ? (
        <>
          <PageHead title={<span className="num">Member {memberId}</span>}>Loading risk profile…</PageHead>
          <div className="grid" style={{ ["--min" as string]: "300px" }}>
            <CardSkeleton h={160} />
            <CardSkeleton h={160} />
            <CardSkeleton h={160} />
          </div>
        </>
      ) : error ? (
        <>
          <PageHead title={<span className="num">Member {memberId}</span>} />
          <Card>
            <ErrorState error={error} what="Member" retry={() => refetch()} />
          </Card>
        </>
      ) : (
        <Profile data={data!} />
      )}
    </>
  );
}

function Profile({ data }: { data: MemberDetail }) {
  const s = data.score;
  const f = data.features ?? {};
  const p = data.policy;
  const toast = useToast();
  const [live, setLive] = useState<"idle" | "loading">("idle");
  const benefit = s.p_churn * p.save_rate * s.clv;
  const maxLogit = Math.max(0.01, ...s.reasons.map((r) => Math.abs(r.logit ?? 0)));

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(s.msno);
      toast({ kind: "success", title: "msno copied", body: s.msno });
    } catch {
      toast({ kind: "error", title: "Clipboard unavailable", body: "Your browser blocked clipboard access." });
    }
  };

  const rescore = async () => {
    setLive("loading");
    try {
      const r = await api.scoreMember(s.member_id, monthStart(data.month));
      const diff = Math.abs(r.p_churn - s.p_churn);
      toast(
        diff < 1e-9
          ? { kind: "success", title: "Live score matches the batch", body: `${pct(r.p_churn, 2)} from model v${r.model.version ?? "?"}` }
          : { kind: "error", title: "Live score differs from the batch", body: `${pct(r.p_churn, 2)} live vs ${pct(s.p_churn, 2)} batch. The model may have changed since scoring.` },
      );
    } catch (e) {
      toast({ kind: "error", title: "Live scoring failed", body: e instanceof ApiError ? e.detail : String(e) });
    } finally {
      setLive("idle");
    }
  };

  const expiry = isoDate(s.last_expire);
  const play = s.contact
    ? {
        title: `Send a retention offer before ${expiry}: ${p.incentive_months === 1 ? "one free month" : `${p.incentive_months} free months`}.`,
        body: "Expected benefit exceeds the cost of the offer, so this member is on the contact list.",
      }
    : s.p_churn >= 0.2
      ? {
          title: "Hold: an offer would cost more than it is expected to save.",
          body: "Risk is high, but p × save rate × CLV does not cover the contact cost plus the free month. A cheaper offer could change that.",
        }
      : { title: "No action needed.", body: "Churn risk is low; this member is expected to renew." };

  const groups: Family[] = ["billing", "engagement", "member"];

  return (
    <>
      <PageHead
        title={
          <span style={{ display: "inline-flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <span className="num">Member {s.member_id}</span>
            <TierBadge tier={s.tier} />
            {s.contact && (
              <span className="badge violet">
                <span className="dot" style={{ color: "var(--violet)" }} />
                On contact list
              </span>
            )}
          </span>
        }
        actions={
          <>
            <button type="button" className="btn" onClick={copy}>
              <IconCopy />
              Copy msno
            </button>
            <button type="button" className="btn" onClick={rescore} disabled={live === "loading"}>
              {live === "loading" ? "Scoring…" : "Re-score live"}
            </button>
          </>
        }
      >
        Scored as of <span className="num">{isoDate(s.as_of_date)}</span> · plan paid through <span className="num">{expiry}</span> · {s.model_name} v{s.model_version} ({s.model_type})
      </PageHead>

      <section className="grid" style={{ ["--min" as string]: "300px" }}>
        <Card className="rise" style={{ alignItems: "center" }}>
          <span className="small muted" style={{ alignSelf: "flex-start" }}>
            Churn probability
          </span>
          <Gauge value={s.p_churn} label={pct(s.p_churn)} color={TIER_COLOR[s.tier]} sub={`EV rank ${int(s.ev_rank)} of ${int(s.contacted_total)} contacts`} />
        </Card>

        <Card className="rise" style={{ animationDelay: "60ms" }}>
          <span className="small muted">Retention decision</span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <span style={{ fontSize: 18, fontWeight: 600 }}>{s.contact ? "Contact" : "Hold"}</span>
            <span className={`num ${s.ev > 0 ? "pos" : "muted"}`}>EV {ntd(s.ev, { signed: true })}</span>
          </div>
          <div className="well">
            <div className="between small">
              <span className="muted num">
                {s.p_churn.toFixed(3)} × {pct(p.save_rate, 0)} × {ntd(s.clv)} CLV
              </span>
              <span className="num">{ntd(benefit)}</span>
            </div>
            <div className="between small">
              <span className="muted">
                NT${p.contact_cost} contact + {p.incentive_months} free month{p.incentive_months === 1 ? "" : "s"}
              </span>
              <span className="num">−{ntd(s.cost)}</span>
            </div>
            <div style={{ height: 1, background: "var(--line)" }} />
            <div className="between" style={{ fontSize: 13, fontWeight: 500 }}>
              <span>Expected value</span>
              <span className={`num ${s.ev > 0 ? "pos" : ""}`}>{ntd(s.ev, { signed: true })}</span>
            </div>
          </div>
          <p className="card-note">The {pct(p.save_rate, 0)} save rate is an assumption: KKBox has no campaign data to measure it.</p>
        </Card>

        <Card className="rise" style={{ animationDelay: "120ms" }}>
          <span className="small muted">Recommended play</span>
          <p style={{ fontSize: 15, fontWeight: 500, lineHeight: 1.5 }}>{play.title}</p>
          <p className="muted" style={{ fontSize: 13, lineHeight: 1.55 }}>
            {play.body}
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: "auto" }}>
            {f.days_to_expiry !== undefined && f.days_to_expiry !== null && (
              <span className="badge">Expires in {int(Number(f.days_to_expiry))} days</span>
            )}
            {f.active_days_30d !== undefined && f.active_days_30d !== null && (
              <span className="badge">Active {int(Number(f.active_days_30d))} of last 30 days</span>
            )}
            {f.last_plan_days ? <span className="badge">{int(Number(f.last_plan_days))}-day plan</span> : null}
          </div>
        </Card>
      </section>

      <Card
        title="What drives this score"
        action={
          <InfoTip label="About reason codes">
            Contribution to the log-odds: coefficient × standardised feature value, relative to an average member. Positive values push the risk up.
          </InfoTip>
        }
      >
        {s.reasons.length === 0 ? (
          <p className="muted small">No reason codes were recorded for this member.</p>
        ) : (
          <div className="barlist">
            {s.reasons.map((r) => (
              <div key={r.feature} className="barlist-row" style={{ gridTemplateColumns: "minmax(0, 280px) minmax(0, 1fr) 72px" }}>
                <div className="stack" style={{ ["--gap" as string]: "2px" }}>
                  <span>{featureLabel(r.feature)}</span>
                  <span className="num small faint">
                    {r.feature}
                    {r.feature in f ? ` = ${formatFeature(r.feature, f[r.feature])}` : ""}
                  </span>
                </div>
                <Track value={Math.abs(r.logit ?? 0) / maxLogit} color={(r.logit ?? 0) >= 0 ? "var(--violet)" : "var(--cyan)"} height={10} />
                <span className="num" style={{ textAlign: "right" }}>
                  {r.logit === null ? "—" : `${r.logit >= 0 ? "+" : "−"}${Math.abs(r.logit).toFixed(2)}`}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {data.features ? (
        <section className="grid" style={{ ["--min" as string]: "300px" }}>
          {groups.map((g) => (
            <Card key={g} title={g === "member" ? "Member" : g === "billing" ? "Billing" : "Listening"}>
              <div>
                {Object.entries(FEATURES)
                  .filter(([, info]) => info.family === g)
                  .map(([name, info]) => (
                    <div className="kv" key={name}>
                      <span>{info.label}</span>
                      <span className="num">{formatFeature(name, f[name])}</span>
                    </div>
                  ))}
              </div>
            </Card>
          ))}
        </section>
      ) : (
        <Card>
          <p className="muted small">The feature snapshot for this month is not available, so model inputs cannot be shown.</p>
        </Card>
      )}

      <Card>
        <div className="between" style={{ alignItems: "center" }}>
          <div className="stack" style={{ ["--gap" as string]: "4px" }}>
            <span className="small muted">Same score from the live API</span>
            <code className="inline">
              GET /members/{s.member_id}/score?as_of={monthStart(data.month)}
            </code>
          </div>
          <span className="small muted">Batch scored {dateTime(s.scored_at)} · batch and API scores are identical by construction</span>
        </div>
      </Card>
    </>
  );
}
