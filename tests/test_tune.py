from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from churnguard.modeling.tune import Fold, cross_validate, rolling_folds, tune

MONTHS = ["2016-10", "2016-11", "2016-12", "2017-01"]


def test_rolling_folds_only_look_forward():
    folds = rolling_folds(MONTHS)
    assert folds == [
        Fold(("2016-10",), "2016-11"),
        Fold(("2016-10", "2016-11"), "2016-12"),
        Fold(("2016-10", "2016-11", "2016-12"), "2017-01"),
    ]
    assert all(max(f.train_months) < f.valid_month for f in folds)


@pytest.fixture
def monthly():
    rng = np.random.default_rng(0)
    n = 8000
    X = pd.DataFrame({"a": rng.normal(size=n).astype("float32"),
                      "b": rng.normal(size=n).astype("float32"),
                      "c": pd.Series(rng.choice(["x", "y"], n)).astype("string")})
    y = pd.Series((rng.random(n) < expit(-2 + 1.5 * X.a)).astype("int8"))
    month = pd.Series(rng.choice(MONTHS, n))
    return X, y, month


def test_cross_validate_oof_is_out_of_time(monthly):
    X, y, month = monthly
    res = cross_validate(X, y, month, rolling_folds(MONTHS), ["a", "b", "c"], ["c"],
                         {"learning_rate": 0.1, "num_leaves": 7})
    assert len(res.fold_log_loss) == 3 and len(res.best_iterations) == 3
    # OOF covers every fold's valid month, never the first month (no history to learn from).
    assert set(res.oof_month) == {"2016-11", "2016-12", "2017-01"}
    assert len(res.oof_raw) == month.isin(["2016-11", "2016-12", "2017-01"]).sum()
    assert res.num_boost_round >= np.mean(res.best_iterations)


def test_tune_runs_and_improves_on_a_bad_default(monthly):
    X, y, month = monthly
    study = tune(X, y, month, rolling_folds(MONTHS), ["a", "b", "c"], ["c"],
                 n_trials=4, seed=0)
    assert len(study.trials) == 4
    assert study.best_value < 0.5  # well under the coin-flip log loss of 0.69
    assert "best_iterations" in study.best_trial.user_attrs
