from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.config import MonitoringConfig
from churnguard.monitoring.monitor import _prev_months, data_quality, decide

CFG = MonitoringConfig()
DQ_OK = {"volume_ok": True, "volume_ratio": 1.0, "p_in_range": True,
         "max_null_jump": {"feature": "x", "delta": 0.0}}


def _drift(*psis):
    return pd.DataFrame({"feature": [f"f{i}" for i in range(len(psis))], "psi": psis})


def _perf(month, status, auc=0.9, ece=0.005):
    return {"month": month, "status": status, "metrics": {"roc_auc": auc, "ece": ece}}


def test_all_quiet_is_ok():
    d = decide(DQ_OK, 0.02, _drift(0.01, 0.05), [_perf("2017-02", "complete")], 0.896, CFG)
    assert d == {"status": "ok", "retrain_reasons": [], "warnings": []}


@pytest.mark.parametrize("score_psi, drift, perf, reason", [
    (0.30, _drift(0.01), [], "score PSI"),
    (0.02, _drift(0.3, 0.4, 0.5), [], "3 features with major drift"),
    (0.02, _drift(0.01), [_perf("2016-12", "complete", auc=0.83)], "2016-12 AUC"),
    (0.02, _drift(0.01), [_perf("2017-01", "complete", ece=0.05)], "ECE"),
])
def test_retrain_triggers(score_psi, drift, perf, reason):
    d = decide(DQ_OK, score_psi, drift, perf, 0.896, CFG)
    assert d["status"] == "retrain" and any(reason in r for r in d["retrain_reasons"])


def test_in_sample_months_never_trigger_retrain():
    perf = [{**_perf("2017-02", "complete", auc=0.70), "in_sample": True}]
    assert decide(DQ_OK, 0.02, _drift(0.01), perf, 0.896, CFG)["status"] == "ok"


def test_partial_labels_never_trigger_retrain():
    # Early-resolving labels are mostly renewals: a bad AUC on them is not evidence.
    d = decide(DQ_OK, 0.02, _drift(0.01), [_perf("2017-03", "partial", auc=0.6)], 0.896, CFG)
    assert d["status"] == "ok"


@pytest.mark.parametrize("dq, score_psi, drift, reason", [
    (DQ_OK, 0.15, _drift(0.01), "score PSI"),
    (DQ_OK, 0.02, _drift(0.4, 0.01), "major drift in f0"),
    ({**DQ_OK, "volume_ok": False, "volume_ratio": 3.0}, 0.02, _drift(0.01), "volume ratio"),
    ({**DQ_OK, "max_null_jump": {"feature": "age", "delta": 0.2}}, 0.02, _drift(0.01),
     "null rate of age"),
])
def test_warnings(dq, score_psi, drift, reason):
    d = decide(dq, score_psi, drift, [], 0.896, CFG)
    assert d["status"] == "warn" and any(reason in w for w in d["warnings"])


def test_data_quality_volume_and_nulls():
    from churnguard.features.registry import numeric_features

    num = numeric_features()
    ref = pd.DataFrame(np.ones((100, len(num))), columns=num)
    cur = ref.iloc[:30].copy()
    cur.loc[:, num[0]] = np.nan
    dq = data_quality(ref, cur, np.array([0.1, 0.5]), ref_rows_per_month=100, cfg=CFG)
    assert dq["volume_ratio"] == 0.3 and not dq["volume_ok"]
    assert dq["max_null_jump"] == {"feature": num[0], "delta": 1.0}


def test_prev_months_crosses_year():
    assert _prev_months("2017-02", 3) == ["2017-01", "2016-12", "2016-11"]


def test_expected_drift_features_do_not_alert():
    # days_since_first_txn is clocked against the dataset start and drifts every month.
    drift = pd.DataFrame({"feature": ["days_since_first_txn", "f1"], "psi": [0.8, 0.01]})
    assert decide(DQ_OK, 0.02, drift, [], 0.896, CFG)["status"] == "warn"
    assert decide(DQ_OK, 0.02, drift, [], 0.896, CFG,
                  expected_drift={"days_since_first_txn"})["status"] == "ok"


def test_registry_marks_the_clock_feature():
    from churnguard.features.registry import expected_drift_features

    assert expected_drift_features() == {"days_since_first_txn"}
