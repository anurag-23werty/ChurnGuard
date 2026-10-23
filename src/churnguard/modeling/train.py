"""Train, calibrate, explain, track and (maybe) promote the LightGBM churn model.

  1. Optuna over rolling folds inside the train months (valid/test untouched)
  2. Re-run CV with the best params -> out-of-time OOF scores -> isotonic calibrator
  3. Final fit on all train rows for mean(best_iteration) x 1.1 rounds, no early stopping
  4. Evaluate raw + calibrated on valid (Feb, derived) and test (Mar, official labels)
  5. SHAP global importance + example reason codes
  6. MLflow: params, metrics, artifacts, registry (challenger) and the promotion gate

Usage:
  python -m churnguard.modeling.train                 # full run
  python -m churnguard.modeling.train --trials 5      # quick run
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import mlflow
import numpy as np

from churnguard import tracking
from churnguard.config import Settings, get_settings
from churnguard.db import connect
from churnguard.features.registry import categorical_features, registry
from churnguard.logging_utils import get_logger, setup_logging
from churnguard.modeling.baselines import check_permutation_control
from churnguard.modeling.dataset import load_split
from churnguard.modeling.gbm import ChurnModel
from churnguard.modeling.metrics import evaluate
from churnguard.modeling.tune import cross_validate, rolling_folds, tune

log = get_logger(__name__)

SHAP_SAMPLE = 50_000
REPORT_METRICS = ("roc_auc", "pr_auc", "log_loss", "brier", "ece", "precision_top10",
                  "recall_top10", "lift_top10")


def _flat(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items() if k in REPORT_METRICS}


def run(settings: Settings, n_trials: int, timeout_s: int | None) -> dict:
    m = settings.modeling
    con = connect(settings)
    train, valid, test = (load_split(settings, s, con) for s in ("train", "valid", "test"))
    features, categorical = list(registry()), categorical_features()
    folds = rolling_folds(m.train_months)
    month = train.meta["label_month"]
    report: dict = {"rows": {"train": len(train), "valid": len(valid), "test": len(test)},
                    "folds": [f.__dict__ for f in folds]}

    t0 = time.monotonic()
    study = tune(train.X, train.y, month, folds, features, categorical, n_trials, m.seed,
                 timeout_s)
    params = {**study.best_params, "bagging_freq": 1}
    report["tuning"] = {"trials": len(study.trials), "best_cv_log_loss": study.best_value,
                        "best_params": params, "seconds": round(time.monotonic() - t0)}
    log.info("tuning: %d trials, best cv log loss %.5f", len(study.trials), study.best_value)

    cv = cross_validate(train.X, train.y, month, folds, features, categorical, params)
    model = ChurnModel(features, categorical, params).fit(
        train.X, train.y, num_boost_round=cv.num_boost_round)
    model.calibrate(cv.oof_raw, cv.oof_y)
    report["cv"] = {"log_loss": cv.log_loss, "fold_log_loss": cv.fold_log_loss,
                    "best_iterations": cv.best_iterations, "num_boost_round": cv.num_boost_round,
                    "oof_rows": len(cv.oof_y)}

    for split in (valid, test):
        raw = model.predict_raw(split.X)
        report[split.name] = {"raw": evaluate(split.y.to_numpy(), raw),
                              "calibrated": evaluate(split.y.to_numpy(),
                                                     model.calibrator.predict(raw))}
        r = report[split.name]["calibrated"]
        log.info("[%s] AUC %.4f PR-AUC %.4f logloss %.4f ECE %.4f (raw ECE %.4f)", split.name,
                 r["roc_auc"], r["pr_auc"], r["log_loss"], r["ece"],
                 report[split.name]["raw"]["ece"])

    sample = valid.X.sample(min(SHAP_SAMPLE, len(valid)), random_state=m.seed)
    importance = model.feature_importance(sample)
    report["shap_importance"] = {k: round(float(v), 5) for k, v in importance.items()}
    top = valid.X.loc[np.argsort(-model.predict_proba(valid.X))[:5]]
    report["example_reason_codes"] = [
        {"member_id": int(valid.meta.loc[i, "member_id"]),
         "churn_probability": round(float(p), 4),
         "reasons": [(f, round(v, 3)) for f, v in rc]}
        for i, p, rc in zip(top.index, model.predict_proba(top), model.reason_codes(top),
                            strict=True)]
    return {"report": report, "model": model, "input_example": valid.X.head(5)}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train the LightGBM churn model")
    parser.add_argument("--trials", type=int, default=settings.modeling.tuning_trials)
    parser.add_argument("--timeout", type=int, default=settings.modeling.tuning_timeout_s)
    parser.add_argument("--no-promote", action="store_true", help="register as challenger only")
    args = parser.parse_args(argv)

    tracking.setup(settings)
    baseline_path = settings.paths.reports / "baselines" / "metrics.json"
    if not baseline_path.exists():
        log.error("run `make baselines` first: the promotion gate compares against it")
        return 1
    baselines = json.loads(baseline_path.read_text())

    with mlflow.start_run(run_name="lgbm") as active:
        run_id = active.info.run_id
        out = run(settings, args.trials, args.timeout)
        report, model = out["report"], out["model"]
        report["run_id"] = run_id

        mlflow.log_params({**report["tuning"]["best_params"],
                           "num_boost_round": report["cv"]["num_boost_round"],
                           "train_rows": report["rows"]["train"],
                           "train_months": ",".join(settings.modeling.train_months),
                           "n_features": len(model.features)})
        mlflow.log_metrics({"cv_log_loss": report["cv"]["log_loss"],
                            **_flat("valid", report["valid"]["calibrated"]),
                            **_flat("test", report["test"]["calibrated"]),
                            **_flat("valid_raw", report["valid"]["raw"])})

        model_dir = settings.paths.models / "gbm" / run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        reports_dir = settings.paths.reports / "gbm"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
        mlflow.log_artifact(str(reports_dir / "metrics.json"))

    version = tracking.log_and_register(settings, model_dir / "model.joblib", run_id,
                                        out["input_example"])
    decision = tracking.promotion_decision(
        report["valid"]["calibrated"], baselines["logreg"]["valid"],
        tracking.champion_metrics(settings),
        permutation_passed=_permutation_passed(baselines))
    log.info("registered %s v%s as challenger; promotion: %s (%s)",
             settings.mlflow.registered_model, version, decision.promote,
             "; ".join(decision.reasons))
    if decision.promote and not args.no_promote:
        tracking.promote(settings, version)
        log.info("v%s is now %s", version, tracking.CHAMPION)
    return 0


def _permutation_passed(baselines: dict) -> bool:
    try:
        check_permutation_control(baselines)
    except (AssertionError, KeyError):
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
