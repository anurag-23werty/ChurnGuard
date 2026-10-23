from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.decisioning.retention import (
    RetentionPolicy,
    expected_value,
    monthly_revenue,
    realised_value,
    retained_months,
    select,
    targeting_curve,
)

POLICY = RetentionPolicy(save_rate=0.2, horizon_months=12, baseline_monthly_churn=0.0,
                         contact_cost=2.0, incentive_months=1.0)


def test_monthly_revenue_normalises_plan_length_and_free_plans():
    rev = monthly_revenue([149, 1788, 0, 99, np.nan], [30, 360, 30, 0, 30])
    assert rev == pytest.approx([149, 149, 0, 0, 0])


def test_retained_months_geometric():
    assert retained_months(POLICY) == 12
    assert retained_months(RetentionPolicy(baseline_monthly_churn=0.5, horizon_months=3)) == 1.75


def test_ev_prefers_value_over_raw_risk():
    # A: very likely to churn but pays nothing. B: less risky, paying member.
    ev = expected_value([0.9, 0.6], [0.0, 149.0], POLICY)
    assert ev.loc[0, "ev"] == pytest.approx(-2.0)  # contact cost only, nothing to save
    assert ev.loc[1, "ev"] == pytest.approx(0.6 * 0.2 * 149 * 12 - (2 + 149))
    assert select(ev).tolist() == [False, True]


def test_break_even_risk_for_a_free_month_offer():
    # p * save_rate * 12 months > 1 free month + contact cost  ->  p > ~0.42 at 149/month
    ev = expected_value([0.41, 0.43], [149.0, 149.0], POLICY)
    assert ev["ev"].tolist()[0] < 0 < ev["ev"].tolist()[1]


def test_select_never_contacts_negative_ev_and_respects_budget():
    ev = expected_value([0.9, 0.8, 0.7, 0.01], [149.0] * 4, POLICY)
    assert select(ev).tolist() == [True, True, True, False]
    one = select(ev, budget=ev.loc[0, "cost"])  # room for exactly one
    assert one.tolist() == [True, False, False, False]


def test_realised_value_counts_only_true_churners():
    ev = expected_value([0.5, 0.5], [149.0, 149.0], POLICY)
    chosen = pd.Series([True, True])
    got = realised_value(ev, chosen, [1, 0], POLICY)
    assert got == pytest.approx(0.2 * 149 * 12 - 2 * (2 + 149))


def test_targeting_curve_ev_beats_risk_when_value_varies():
    rng = np.random.default_rng(0)
    n = 20_000
    p = rng.beta(1, 12, n)
    rev = rng.choice([0.0, 99.0, 149.0, 300.0], n, p=[0.3, 0.3, 0.3, 0.1])
    y = (rng.random(n) < p).astype(int)
    policy = RetentionPolicy(save_rate=0.3, baseline_monthly_churn=0.05)
    curve = targeting_curve(expected_value(p, rev, policy), y, policy, fractions=(0.05,))
    assert curve.loc[0, "by_ev"] > curve.loc[0, "by_risk"] > curve.loc[0, "random"]
