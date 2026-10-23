from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from churnguard.config import PROJECT_ROOT, get_settings
from churnguard.console import data

APP = str(PROJECT_ROOT / "src" / "churnguard" / "console" / "app.py")


def _scores(n=50):
    rng = np.random.default_rng(0)
    ev = np.sort(rng.normal(-20, 60, n))[::-1]
    df = pd.DataFrame({
        "ev_rank": np.arange(1, n + 1), "member_id": np.arange(100, 100 + n),
        "msno": [f"m{i}" for i in range(n)], "p_churn": rng.random(n),
        "monthly_revenue": 149.0, "clv": 1389.0, "cost": 151.0, "ev": ev,
        "contact": ev > 0, "reason_1": rng.choice(["last_is_cancel", "last_is_auto_renew"], n),
        "reason_2": "days_to_expiry", "reason_3": None, "model_name": "churnguard",
        "model_version": "1", "model_type": "logreg", "scored_at": "2017-04-01T00:00:00",
    })
    return df


@pytest.fixture
def scored(settings):
    d = settings.paths.gold / "scores" / "score_month=2017-04"
    d.mkdir(parents=True)
    _scores().to_parquet(d / "data.parquet")
    rep = settings.paths.reports / "monitoring"
    rep.mkdir(parents=True)
    (rep / "month=2017-04.json").write_text(json.dumps({
        "decision": {"status": "warn", "retrain_reasons": [], "warnings": ["score PSI 0.12"]},
        "score_drift": {"psi": 0.12, "severity": "moderate"},
        "data_quality": {"volume_ratio": 1.03, "max_null_jump": {"feature": "age",
                                                                  "delta": 0.01}},
        "feature_drift": [{"feature": "n_txn_30d", "psi": 0.17, "severity": "moderate"}],
        "performance": [{"month": "2017-04", "status": "pending", "decided_share": 0.0,
                         "in_sample": False}]}))
    return settings


def test_summary_curve_budget_and_export(scored):
    s = data.load_scores(scored, "2017-04")
    summ = data.campaign_summary(s)
    assert summ["contacted"] == int((s.ev > 0).sum())
    assert summ["expected_net_value"] == pytest.approx(s.ev[s.ev > 0].sum())
    curve = data.value_curve(s)
    assert curve.cumulative_expected_value.idxmax() == curve.contacted.sub(
        summ["contacted"]).abs().idxmin()  # peak where EV turns negative
    two = data.budget_selection(s, budget=2 * 151.0)
    assert two.contact.sum() == 2 and two.loc[two.contact, "ev_rank"].tolist() == [1, 2]
    csv = data.crm_export(s).decode().splitlines()
    assert csv[0].startswith("ev_rank,msno") and len(csv) == summ["contacted"] + 1
    assert data.reason_mix(s).share.sum() == pytest.approx(1.0)


def test_find_member_by_msno_or_id(scored):
    s = data.load_scores(scored, "2017-04")
    assert data.find_member(s, "m3").member_id.item() == 103
    assert data.find_member(s, "104").msno.item() == "m4"
    assert data.find_member(s, "nobody").empty and data.find_member(s, " ").empty


def test_console_app_renders_every_tab(scored, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    cfg = tmp_path / "console.yaml"
    p = scored.paths
    cfg.write_text(f"""
paths: {{raw: {p.raw}, bronze: {p.bronze}, silver: {p.silver}, gold: {p.gold},
        reports: {p.reports}, models: {p.models}}}
kaggle: {{competition: test}}
""")
    monkeypatch.setenv("CHURNGUARD_CONFIG", str(cfg))
    get_settings.cache_clear()
    try:
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception
        assert at.title[0].value == "ChurnGuard retention console"
        metrics = {m.label: m.value for m in at.metric}
        contacted = int((_scores().ev > 0).sum())
        assert metrics["Members due"] == "50" and metrics["Contact (EV > 0)"] == str(contacted)
        assert at.warning[0].value.startswith("Decision: WARN")
        at.text_input[0].input("m0").run()
        assert {m.label: m.value for m in at.metric}["Decision"] == "contact"
    finally:
        get_settings.cache_clear()


def test_console_app_without_scores(settings, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    cfg = tmp_path / "empty.yaml"
    p = settings.paths
    cfg.write_text(f"""
paths: {{raw: {p.raw}, bronze: {p.bronze}, silver: {p.silver}, gold: {p.gold},
        reports: {p.reports}, models: {p.models}}}
kaggle: {{competition: test}}
""")
    monkeypatch.setenv("CHURNGUARD_CONFIG", str(cfg))
    get_settings.cache_clear()
    try:
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception and "make score" in at.info[0].value
    finally:
        get_settings.cache_clear()
