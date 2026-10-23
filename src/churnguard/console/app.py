"""ChurnGuard retention console.

Run:  make console   (streamlit run src/churnguard/console/app.py)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from churnguard.config import get_settings
from churnguard.console import data

st.set_page_config(page_title="ChurnGuard", layout="wide")
settings = get_settings()

months = data.scored_months(settings)
st.title("ChurnGuard retention console")
if not months:
    st.info("No scored months yet. Run `make score` first.")
    st.stop()

month = st.sidebar.selectbox("Score month", months)
scores = data.load_scores(settings, month)
summary = data.campaign_summary(scores)
st.sidebar.caption(f"Model: {summary['model']}")
st.sidebar.caption(f"Save-rate assumption: {settings.retention.save_rate:.0%} "
                   "(no campaign data yet: measure with a randomised holdout)")

campaign, budget_tab, monitoring_tab, lookup = st.tabs(
    ["Campaign", "Budget", "Monitoring", "Member lookup"])

with campaign:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Members due", f"{summary['members']:,}")
    c2.metric("Contact (EV > 0)", f"{summary['contacted']:,}",
              f"{summary['contact_share']:.1%} of due")
    c3.metric("Expected net value", f"{summary['expected_net_value']:,.0f} NTD")
    c4.metric("Mean churn risk", f"{summary['mean_p_churn']:.1%}")

    st.subheader("Contact list")
    shown = scores[scores.contact][["ev_rank", "msno", "p_churn", "monthly_revenue", "ev",
                                    "reason_1", "reason_2", "reason_3"]]
    st.dataframe(shown.head(500), hide_index=True, width="stretch",
                 column_config={"p_churn": st.column_config.NumberColumn(format="%.3f"),
                                "ev": st.column_config.NumberColumn(format="%.1f")})
    st.download_button("Download CRM list (CSV)", data.crm_export(scores),
                       file_name=f"churnguard_contacts_{month}.csv", mime="text/csv")
    st.subheader("Why they are at risk (top reason)")
    st.bar_chart(data.reason_mix(scores), x="reason", y="share")

with budget_tab:
    full_cost = float(scores.loc[scores.ev > 0, "cost"].sum())
    budget = st.slider("Campaign budget (NTD)", 0.0, max(full_cost, 1.0), full_cost,
                       step=max(full_cost / 100, 1.0))
    chosen = data.budget_selection(scores, budget)
    s = data.campaign_summary(chosen)
    b1, b2 = st.columns(2)
    b1.metric("Contacted", f"{s['contacted']:,}")
    b2.metric("Expected net value", f"{s['expected_net_value']:,.0f} NTD")
    st.caption("Cumulative expected value by number contacted (EV order): the peak is where "
               "the next offer costs more than the churn it is expected to save.")
    st.line_chart(data.value_curve(scores), x="contacted", y="cumulative_expected_value")

with monitoring_tab:
    report = data.monitoring_report(settings, month)
    if report is None:
        st.info(f"No monitoring report for {month}. Run `make monitor MONTH={month}`.")
    else:
        d = report["decision"]
        {"ok": st.success, "warn": st.warning, "retrain": st.error}[d["status"]](
            f"Decision: {d['status'].upper()}  "
            + "; ".join(d["retrain_reasons"] or d["warnings"] or ["nothing material"]))
        m1, m2, m3 = st.columns(3)
        m1.metric("Score PSI", f"{report['score_drift']['psi']:.3f}",
                  report["score_drift"]["severity"])
        m2.metric("Volume vs reference", f"{report['data_quality']['volume_ratio']:.2f}x")
        m3.metric("Max null-rate jump",
                  f"{report['data_quality']['max_null_jump']['delta']:.3f}",
                  report["data_quality"]["max_null_jump"]["feature"])
        st.subheader("Feature drift (PSI)")
        st.dataframe(pd.DataFrame(report["feature_drift"]).head(15), hide_index=True,
                     width="stretch")
        st.subheader("Performance as labels mature")
        st.dataframe(pd.DataFrame([{**{k: p.get(k) for k in ("month", "status",
                                                             "decided_share", "in_sample")},
                                    **p.get("metrics", {})} for p in report["performance"]]),
                     hide_index=True, width="stretch")

with lookup:
    query = st.text_input("msno or member_id")
    hit = data.find_member(scores, query)
    if query and hit.empty:
        st.warning(f"{query} is not due in {month}.")
    elif not hit.empty:
        r = hit.iloc[0]
        l1, l2, l3 = st.columns(3)
        l1.metric("Churn risk", f"{r.p_churn:.1%}")
        l2.metric("Expected value of contacting", f"{r.ev:,.1f} NTD")
        l3.metric("Decision", "contact" if r.contact else "do not contact")
        st.write("Top reasons:", ", ".join(x for x in (r.reason_1, r.reason_2, r.reason_3)
                                           if isinstance(x, str)) or "none")
