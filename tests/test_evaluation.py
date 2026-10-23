from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from churnguard.modeling.candidates import SklearnChurnModel
from churnguard.modeling.evaluation import calibration_helps, out_of_time, summarise
from churnguard.modeling.preprocessing import make_logreg

MONTHS = ["2016-10", "2016-11", "2016-12", "2017-01"]


@pytest.fixture
def monthly():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"a": rng.normal(size=n).astype("float32"),
                      "c": pd.Series(rng.choice(["x", "y"], n)).astype("string")})
    y = pd.Series((rng.random(n) < expit(-2 + 1.5 * X.a)).astype("int8"))
    return X, y, pd.Series(rng.choice(MONTHS, n))


class Constant:
    features = ["a"]

    def fit(self, X, y):
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate)


def test_out_of_time_scores_every_month_after_the_first(monthly):
    X, y, month = monthly
    cands = {"lr": lambda: SklearnChurnModel(make_logreg(["a"], ["c"]), ["a", "c"]),
             "const": Constant}
    per_month, oof = out_of_time(cands, X, y, month, MONTHS)
    assert sorted(per_month.month.unique()) == MONTHS[1:]
    assert len(per_month) == 2 * 3 and set(oof) == {"lr", "const"}
    s = summarise(per_month, baseline="const")
    assert s.index[0] == "lr"
    assert s.loc["lr", "months_beating_baseline_pr_auc"] == 3
    assert s.loc["const", "worst_roc_auc"] == pytest.approx(0.5)


def test_calibration_applied_only_when_it_helps():
    rng = np.random.default_rng(1)
    p = rng.random(30_000)
    y = (rng.random(len(p)) < p).astype(int)
    months = np.repeat(["m1", "m2", "m3"], 10_000)
    good = pd.DataFrame({"month": months, "p": p, "y": y})
    assert calibration_helps(good)[0] is False  # already calibrated: leave it alone
    bad = good.assign(p=p ** 3)  # systematically under-predicts
    helps, detail = calibration_helps(bad)
    assert helps and detail["cal_ece"] < detail["raw_ece"]
