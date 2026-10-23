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

BASE = {"mean_pr_auc": 0.696, "mean_log_loss": 0.132, "worst_roc_auc": 0.832}


def _cand(**kw):
    return {"mean_pr_auc": 0.72, "mean_log_loss": 0.128, "worst_roc_auc": 0.83, **kw}


def test_gate_promotes_a_clear_improvement():
    d = tracking.promotion_decision(_cand(), BASE, 0.004, champion=None,
                                    permutation_passed=True)
    assert d.promote and d.reasons == ["all gates passed"]


@pytest.mark.parametrize("cand, ece, champion, perm, reason", [
    (_cand(mean_pr_auc=0.699), 0.004, None, True, "mean PR-AUC 0.6990 < baseline"),
    (_cand(mean_log_loss=0.14), 0.004, None, True, "mean log loss"),
    # The real LightGBM v1 case: great on Feb, but Dec 2016 AUC 0.781 vs baseline 0.832.
    (_cand(worst_roc_auc=0.781), 0.004, None, True, "worst-month AUC"),
    (_cand(), 0.03, None, True, "ECE"),
    (_cand(), 0.004, {"mean_pr_auc": 0.73}, True, "champion"),
    (_cand(), 0.004, None, False, "permutation control"),
])
def test_gate_blocks(cand, ece, champion, perm, reason):
    d = tracking.promotion_decision(cand, BASE, ece, champion, perm)
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
        mlflow.log_metric("oot_mean_pr_auc", 0.7)
    version = tracking.log_and_register(s, path, run.info.run_id, X.head(3))
    tracking.promote(s, version)

    assert tracking.champion_metrics(s) == {"mean_pr_auc": 0.7}
    served = mlflow.pyfunc.load_model(f"models:/m@{tracking.CHAMPION}")
    out = served.predict(X.head(4))
    assert list(out.columns) == ["churn_probability"]
    assert out.churn_probability.to_numpy() == pytest.approx(model.predict_proba(X.head(4)))
    loaded = tracking.load_champion(s)
    assert isinstance(loaded, ChurnModel)
    assert loaded.predict_proba(X.head(4)) == pytest.approx(model.predict_proba(X.head(4)))


def test_demote_removes_alias(settings, tmp_path):
    s = replace(settings, mlflow=MLflowConfig(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
                                              experiment="t2", registered_model="m2"))
    tracking.setup(s, artifact_root=tmp_path / "artifacts")
    X = pd.DataFrame({"a": np.arange(200, dtype="float32"),
                      "c": pd.Series(["x", "y"] * 100).astype("string")})
    model = ChurnModel(["a", "c"], ["c"], {"num_leaves": 4}).fit(
        X, (X.a > 100).astype("int8"), num_boost_round=5)
    joblib.dump(model, tmp_path / "m.joblib")
    with mlflow.start_run() as run:
        pass
    v = tracking.log_and_register(s, tmp_path / "m.joblib", run.info.run_id, X.head(2),
                                  model_type="lgbm")
    tracking.promote(s, v)
    assert tracking.demote("m2") is True
    assert tracking.champion_metrics(s) is None
    assert tracking.demote("m2") is False
    assert mlflow.MlflowClient().get_model_version("m2", v).tags["model_type"] == "lgbm"


def test_load_champion_from_read_only_artifact_store(settings, tmp_path):
    """Regression: the Docker container mounts mlartifacts read-only, and loading
    models:/...@champion used to write registered_model_meta into it (EROFS -> HTTP 500)."""
    import os
    import stat

    s = replace(settings, mlflow=MLflowConfig(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
                                              experiment="ro", registered_model="ro"))
    artifacts = tmp_path / "artifacts"
    tracking.setup(s, artifact_root=artifacts)
    X = pd.DataFrame({"a": np.arange(200, dtype="float32"),
                      "c": pd.Series(["x", "y"] * 100).astype("string")})
    model = ChurnModel(["a", "c"], ["c"], {"num_leaves": 4}).fit(
        X, (X.a > 100).astype("int8"), num_boost_round=5)
    joblib.dump(model, tmp_path / "m.joblib")
    with mlflow.start_run() as run:
        pass
    tracking.promote(s, tracking.log_and_register(s, tmp_path / "m.joblib", run.info.run_id,
                                                  X.head(2)))
    paths = [artifacts, *artifacts.rglob("*")]
    for p in paths:
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        loaded = tracking.load_champion(s)
        assert loaded.predict_proba(X.head(3)) == pytest.approx(model.predict_proba(X.head(3)))
        assert not any(p.name == "registered_model_meta" for p in artifacts.rglob("*"))
    finally:
        for p in paths:
            os.chmod(p, p.stat().st_mode | stat.S_IWUSR)
