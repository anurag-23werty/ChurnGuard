from __future__ import annotations

from dataclasses import replace

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from churnguard import tracking
from churnguard.config import MLflowConfig
from churnguard.modeling.gbm import ChurnModel

BASE = {"pr_auc": 0.61, "log_loss": 0.109, "ece": 0.006}


def _cand(**kw):
    return {"pr_auc": 0.65, "log_loss": 0.100, "ece": 0.004, **kw}


def test_gate_promotes_a_clear_improvement():
    d = tracking.promotion_decision(_cand(), BASE, champion=None, permutation_passed=True)
    assert d.promote and d.reasons == ["all gates passed"]


@pytest.mark.parametrize("cand, champion, perm, reason", [
    (_cand(pr_auc=0.615), None, True, "PR-AUC 0.6150 < baseline"),
    (_cand(log_loss=0.12), None, True, "log loss"),
    (_cand(ece=0.03), None, True, "ECE"),
    (_cand(), {"pr_auc": 0.66}, True, "champion"),
    (_cand(), None, False, "permutation control"),
])
def test_gate_blocks(cand, champion, perm, reason):
    d = tracking.promotion_decision(cand, BASE, champion, perm)
    assert not d.promote and any(reason in r for r in d.reasons)


def test_register_alias_and_load_champion(settings, tmp_path):
    s = replace(settings, mlflow=MLflowConfig(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
                                              experiment="t", registered_model="m"))
    tracking.setup(s, artifact_root=tmp_path / "artifacts")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500).astype("float32"),
                      "c": pd.Series(rng.choice(["x", "y"], 500)).astype("string")})
    y = (X.a > 0.5).astype("int8")
    model = ChurnModel(["a", "c"], ["c"], {"num_leaves": 4}).fit(X, y, num_boost_round=10)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    assert tracking.champion_metrics(s) is None
    with mlflow.start_run() as run:
        mlflow.log_metric("valid_pr_auc", 0.7)
    version = tracking.log_and_register(s, path, run.info.run_id, X.head(3))
    tracking.promote(s, version)

    assert tracking.champion_metrics(s) == {"pr_auc": 0.7}
    served = mlflow.pyfunc.load_model(f"models:/m@{tracking.CHAMPION}")
    out = served.predict(X.head(4))
    assert list(out.columns) == ["churn_probability"]
    assert out.churn_probability.to_numpy() == pytest.approx(model.predict_proba(X.head(4)))
    loaded = tracking.load_champion(s)
    assert isinstance(loaded, ChurnModel)
    assert loaded.predict_proba(X.head(4)) == pytest.approx(model.predict_proba(X.head(4)))
