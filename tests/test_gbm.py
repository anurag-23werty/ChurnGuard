from __future__ import annotations

import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from churnguard.modeling.gbm import ChurnModel


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({
        "auto_renew": rng.integers(0, 2, n).astype("float32"),
        "secs": rng.exponential(1000, n).astype("float32"),
        "noise": rng.normal(size=n).astype("float32"),
        "city": pd.Series(rng.choice(["1", "5", "13"], n)).astype("string"),
    })
    X.loc[::9, "secs"] = np.nan
    is13 = (X.city == "13").astype(float)
    z = -1.5 - 2.5 * X.auto_renew + 1.0 * is13 - 0.0005 * X.secs.fillna(1000)
    y = pd.Series((rng.random(n) < expit(z)).astype("int8"))
    return X, y


def _model(**params):
    return ChurnModel(features=["auto_renew", "secs", "noise", "city"], categorical=["city"],
                      params={"learning_rate": 0.1, "num_leaves": 15, **params})


def test_fit_with_early_stopping_and_predict(data):
    X, y = data
    m = _model().fit(X[:4000], y[:4000], X[4000:], y[4000:])
    assert 0 < m.best_iteration < 2000
    p = m.predict_proba(X[4000:])
    assert ((p > 0) & (p < 1)).all()


def test_unseen_and_missing_categories_score(data):
    X, y = data
    m = _model().fit(X, y, num_boost_round=50)
    new = X.head(3).copy()
    new["city"] = pd.Series(["999", pd.NA, "13"], dtype="string")
    assert np.isfinite(m.predict_proba(new)).all()
    assert m.levels["city"] == ["1", "13", "5"]


def test_shap_rows_sum_to_raw_logit(data):
    X, y = data
    m = _model().fit(X, y, num_boost_round=50)
    sv = m.shap_values(X.head(20))
    assert sv.sum(axis=1).to_numpy() == pytest.approx(logit(m.predict_raw(X.head(20))), abs=1e-6)


def test_importance_and_reason_codes_find_the_signal(data):
    X, y = data
    m = _model().fit(X, y, num_boost_round=100)
    imp = m.feature_importance(X)
    assert imp.index[0] == "auto_renew" and imp["noise"] == imp.min()
    # Members without auto-renew: auto_renew should be among the top reasons to churn.
    reasons = m.reason_codes(X[X.auto_renew == 0].head(50), k=2)
    assert sum(any(f == "auto_renew" for f, _ in r) for r in reasons) >= 45
    assert all(v > 0 for r in reasons for _, v in r)


def test_isotonic_calibration_fixes_a_miscalibrated_score(data):
    X, y = data
    m = _model().fit(X[:3000], y[:3000], num_boost_round=50)
    raw = m.predict_raw(X[3000:])
    m.calibrate(raw ** 0.5, y[3000:])  # pretend the booster were badly calibrated
    cal = m.calibrator.predict(raw ** 0.5)
    assert abs(cal.mean() - y[3000:].mean()) < 0.005
    assert m.fit(X, y, num_boost_round=5).calibrator is None  # refit drops calibration


def test_pickled_model_loads_in_fresh_process(data, tmp_path):
    X, y = data
    path = tmp_path / "m.joblib"
    joblib.dump(_model().fit(X, y, num_boost_round=20), path)
    X.head(2).to_parquet(tmp_path / "x.parquet")
    code = (f"import joblib, pandas as pd; m = joblib.load({str(path)!r}); "
            f"x = pd.read_parquet({str(tmp_path / 'x.parquet')!r}); "
            "print(m.predict_proba(x).round(6).tolist())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
