"""Choose, fit, register and (maybe) promote the champion model.

  1. Every candidate (logreg, lgbm, lgbm_robust, blends) on every out-of-time month
     Nov 2016 .. Feb 2017, each trained only on earlier months
  2. Gate v2 per candidate (mean + worst month vs logreg); pick the best that passes
  3. Isotonic calibration only if it helps out of time
  4. Final fit on all history through Feb (Feb subsampled to an average month's size),
     scored on the March test overall and without the early-renewer segment (official
     labels there depend on April events we cannot see)
  5. MLflow: per-month table, summaries, test metrics; register; promote if gate passes

Usage:
  python -m churnguard.modeling.select
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import mlflow
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from churnguard import tracking
from churnguard.config import get_settings
from churnguard.db import connect
from churnguard.logging_utils import get_logger, setup_logging
from churnguard.modeling.baselines import check_permutation_control
from churnguard.modeling.candidates import CalibratedModel, make_candidates
from churnguard.modeling.dataset import load_split
from churnguard.modeling.evaluation import calibration_helps, out_of_time, summarise
from churnguard.modeling.metrics import evaluate

log = get_logger(__name__)

BASELINE = "logreg"
EARLY_RENEWER_DAYS = 31  # days_to_expiry beyond the label month: renewed before M


def _permutation_passed(settings) -> bool:
    path = settings.paths.reports / "baselines" / "metrics.json"
    try:
        check_permutation_control(json.loads(path.read_text()))
    except (AssertionError, KeyError, FileNotFoundError):
        return False
    return True


def _final_training_set(train, valid, seed: int):
    """Train sample + Feb subsampled to an average training month, so Feb does not
    dominate the fit."""
    per_month = int(train.meta.label_month.value_counts().mean())
    idx = valid.y.sample(min(per_month, len(valid)), random_state=seed).index
    X = pd.concat([train.X, valid.X.loc[idx]], ignore_index=True)
    y = pd.concat([train.y, valid.y.loc[idx]], ignore_index=True)
    return X, y


def _test_report(model, test) -> dict:
    p = model.predict_proba(test.X)
    core = (test.X["days_to_expiry"] <= EARLY_RENEWER_DAYS).to_numpy()
    y = test.y.to_numpy()
    return {"all": evaluate(y, p), "excluding_early_renewers": evaluate(y[core], p[core]),
            "early_renewer_share": float(1 - core.mean())}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args(argv)
    m = settings.modeling

    tuned_path = settings.paths.reports / "gbm" / "metrics.json"
    if not tuned_path.exists():
        log.error("run `make train` first (tuned LightGBM params)")
        return 1
    tuned = json.loads(tuned_path.read_text())
    candidates = make_candidates(tuned["tuning"]["best_params"],
                                 tuned["cv"]["num_boost_round"], m.seed)

    con = connect(settings)
    train, valid, test = (load_split(settings, s, con) for s in ("train", "valid", "test"))
    X = pd.concat([train.X, valid.X], ignore_index=True)
    y = pd.concat([train.y, valid.y], ignore_index=True)
    month = pd.concat([train.meta.label_month, valid.meta.label_month], ignore_index=True)
    months = [*m.train_months, m.valid_month]

    per_month, oof = out_of_time(candidates, X, y, month, months)
    summary = summarise(per_month, BASELINE)
    latest = per_month[per_month.month == months[-1]].set_index("candidate")
    perm = _permutation_passed(settings)
    tracking.setup(settings)
    champion = tracking.champion_metrics(settings)

    decisions = {}
    for name in summary.index:
        if name == BASELINE:
            continue
        decisions[name] = tracking.promotion_decision(
            summary.loc[name].to_dict(), summary.loc[BASELINE].to_dict(),
            float(latest.loc[name, "ece"]), champion, perm)
        log.info("[gate] %-20s promote=%s  %s", name, decisions[name].promote,
                 "; ".join(decisions[name].reasons))
    passing = [n for n in summary.index if n in decisions and decisions[n].promote]
    chosen = passing[0] if passing else BASELINE
    log.info("selected: %s (%s)", chosen, "passes gate" if passing
             else "no candidate beat the baseline: logreg stays the model")

    helps, cal_detail = calibration_helps(oof[chosen])
    X_fin, y_fin = _final_training_set(train, valid, m.seed)
    model = candidates[chosen]().fit(X_fin, y_fin)
    baseline_model = candidates[BASELINE]().fit(X_fin, y_fin)
    if helps:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        model = CalibratedModel(model, iso.fit(oof[chosen].p, oof[chosen].y))

    report = {
        "candidates": list(candidates), "scored_months": months[1:],
        "per_month": per_month.round(5).to_dict(orient="records"),
        "summary": summary.round(5).reset_index().to_dict(orient="records"),
        "gate": {k: {"promote": d.promote, "reasons": d.reasons} for k, d in decisions.items()},
        "selected": chosen, "calibration": {"applied": helps, **cal_detail},
        "final_training_rows": len(y_fin),
        "test": {chosen: _test_report(model, test),
                 f"{BASELINE} (same training data)": _test_report(baseline_model, test)},
    }
    out = settings.paths.reports / "selection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(report, indent=2, default=str))
    per_month.to_csv(out / "per_month.csv", index=False)

    with mlflow.start_run(run_name=f"select-{chosen}") as run:
        mlflow.log_params({"selected": chosen, "calibrated": helps,
                           "final_training_rows": len(y_fin)})
        mlflow.log_metrics({f"oot_{k}": float(v) for k, v in summary.loc[chosen].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in report["test"][chosen]["all"].items()
                            if isinstance(v, float)})
        mlflow.log_artifact(str(out / "selection.json"))
        mlflow.log_artifact(str(out / "per_month.csv"))
    model_dir = settings.paths.models / "selected" / run.info.run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    version = tracking.log_and_register(settings, model_dir / "model.joblib", run.info.run_id,
                                        valid.X.head(5), model_type=chosen)
    promote = chosen == BASELINE and champion is None or (
        chosen != BASELINE and decisions[chosen].promote)
    if promote and not args.no_promote:
        tracking.promote(settings, version)
    log.info("registered %s v%s (%s)%s", settings.mlflow.registered_model, version, chosen,
             " -> champion" if promote and not args.no_promote else " as challenger")
    print(summary.round(4).to_string())
    for name, r in report["test"].items():
        print(f"test {name:32} all AUC {r['all']['roc_auc']:.4f} PR {r['all']['pr_auc']:.4f} | "
              f"excl. early renewers AUC {r['excluding_early_renewers']['roc_auc']:.4f} "
              f"PR {r['excluding_early_renewers']['pr_auc']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
