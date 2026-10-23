from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from churnguard.modeling.candidates import BlendModel, SklearnChurnModel
from churnguard.modeling.gbm import ChurnModel
from churnguard.modeling.preprocessing import make_logreg


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame({"renew": rng.random(n).astype("float32"),
                      "noise": rng.normal(size=n).astype("float32"),
                      "c": pd.Series(rng.choice(["a", "b"], n)).astype("string")})
    y = pd.Series((rng.random(n) < expit(1 - 4 * X.renew + 0.3 * X.noise)).astype("int8"))
    return X, y


def test_monotone_constraint_is_enforced(data):
    X, y = data
    m = ChurnModel(["renew", "noise", "c"], ["c"], {"num_leaves": 31},
                   num_boost_round=80, monotone={"renew": -1}).fit(X, y)
    grid = pd.DataFrame({"renew": np.linspace(0, 1, 50, dtype="float32"),
                         "noise": np.float32(0), "c": pd.Series(["a"] * 50, dtype="string")})
    p = m.predict_proba(grid)
    assert (np.diff(p) <= 1e-12).all()


def test_monotone_rejects_unknown_features(data):
    X, y = data
    with pytest.raises(ValueError, match="unknown features"):
        ChurnModel(["renew"], [], monotone={"nope": 1}).fit(X, y)


def test_sklearn_wrapper_and_blend_share_one_interface(data):
    X, y = data
    lr = SklearnChurnModel(make_logreg(["renew", "noise"], ["c"]), ["renew", "noise", "c"])
    gbm = ChurnModel(["renew", "noise", "c"], ["c"], {"num_leaves": 7}, num_boost_round=30)
    blend = BlendModel([gbm, lr], [0.25, 0.75]).fit(X, y)
    p = blend.predict_proba(X.head(10))
    assert p.shape == (10,)
    assert p == pytest.approx(0.25 * gbm.predict_proba(X.head(10))
                              + 0.75 * lr.predict_proba(X.head(10)))
    assert len(blend.reason_codes(X.head(3))) == 3  # delegated to the tree member


def test_blend_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        BlendModel([object(), object()], [0.5, 0.6])


def test_logreg_contributions_sum_to_logit_and_name_input_features(data):
    from scipy.special import logit

    X, y = data
    X = X.assign(renew_rate=X.renew)  # a feature whose name extends another's
    feats = ["renew", "noise", "c", "renew_rate"]
    lr = SklearnChurnModel(make_logreg(["renew", "noise", "renew_rate"], ["c"]), feats)
    lr.fit(X, y)
    contrib = lr.contributions(X.head(50))
    assert list(contrib.columns) == feats
    intercept = lr.pipeline.named_steps["clf"].intercept_[0]
    assert (contrib.sum(axis=1) + intercept).to_numpy() == pytest.approx(
        logit(lr.predict_proba(X.head(50))), abs=1e-6)
    # Low renew pushes towards churn: it should lead the reasons for low-renew members.
    low = X[X.renew < 0.1].head(20)
    assert all(r and r[0][0] in {"renew", "renew_rate"} for r in lr.reason_codes(low))
