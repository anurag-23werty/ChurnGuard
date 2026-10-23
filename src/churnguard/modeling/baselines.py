"""Baselines every later model has to beat, plus the shuffled-label leakage control.

  prior    training churn rate for everyone (log-loss floor)
  rule     churn rate per (last_is_auto_renew, last_is_cancel) cell: the analyst's rule
  logreg   signed-log1p + standardise numerics, missing indicators, one-hot categoricals
  shuffled permutation control: logreg on K label permutations of a train sample

On the permutation control: a model fit on permuted labels has random coefficients, and a
random linear combination of strongly predictive features projects onto the real signal by
chance, in a random direction. So a single permuted model can land well away from 0.5
(0.42-0.59 on KKBox), and a one-seed "AUC must be 0.5 +/- 0.02" check raises false alarms.
The control therefore checks the *distribution*: the mean over K permutations must be within
3 standard errors of 0.5, and the real model must beat every permuted one. A genuine leak
(a feature or preprocessing step carrying label information) shows up as permuted AUCs
that are consistently high with small spread, which fails both.

Usage:
  python -m churnguard.modeling.baselines
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from churnguard.config import Settings, get_settings
from churnguard.db import connect
from churnguard.features.registry import categorical_features, numeric_features
from churnguard.logging_utils import get_logger, setup_logging
from churnguard.modeling.dataset import load_split
from churnguard.modeling.metrics import evaluate
from churnguard.modeling.preprocessing import make_logreg

log = get_logger(__name__)

PERMUTATIONS = 5
PERMUTATION_SAMPLE = 200_000
RULE_COLS = ("last_is_auto_renew", "last_is_cancel")


class PriorModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class RuleModel(BaseEstimator, ClassifierMixin):
    """Smoothed churn rate per cell of `cols`; unseen cells fall back to the prior."""

    def __init__(self, cols: tuple[str, ...] = RULE_COLS, smoothing: float = 10.0):
        self.cols = cols
        self.smoothing = smoothing

    def fit(self, X, y):
        self.prior_ = float(np.mean(y))
        df = X[list(self.cols)].astype("float64").fillna(-1).assign(_y=np.asarray(y))
        g = df.groupby(list(self.cols))["_y"].agg(["sum", "count"])
        self.table_ = ((g["sum"] + self.smoothing * self.prior_)
                       / (g["count"] + self.smoothing)).rename("p")
        return self

    def predict_proba(self, X):
        key = X[list(self.cols)].astype("float64").fillna(-1)
        p = key.merge(self.table_, left_on=list(self.cols), right_index=True, how="left")["p"]
        p = p.fillna(self.prior_).to_numpy()
        return np.column_stack([1 - p, p])


def _fit_eval(name, model, train, evals) -> dict:
    t0 = time.monotonic()
    model.fit(train.X, train.y)
    fit_s = time.monotonic() - t0
    out = {"fit_seconds": round(fit_s, 1)}
    for split in evals:
        out[split.name] = evaluate(split.y.to_numpy(), model.predict_proba(split.X)[:, 1])
    log.info("[%s] fit %.1fs | valid AUC %.4f PR-AUC %.4f logloss %.4f | test AUC %.4f",
             name, fit_s, out["valid"]["roc_auc"], out["valid"]["pr_auc"],
             out["valid"]["log_loss"], out["test"]["roc_auc"])
    return out


def run(settings: Settings) -> dict:
    con = connect(settings)
    train, valid, test = (load_split(settings, s, con) for s in ("train", "valid", "test"))
    log.info("train %s rows (%.2f%% churn) | valid %s (%.2f%%) | test %s (%.2f%%, official)",
             f"{len(train):,}", 100 * train.y.mean(), f"{len(valid):,}", 100 * valid.y.mean(),
             f"{len(test):,}", 100 * test.y.mean())
    num, cat = numeric_features(), categorical_features()
    seed = settings.modeling.seed

    results: dict = {}
    models = {
        "prior": PriorModel(),
        "rule": RuleModel(),
        "logreg": make_logreg(num, cat, seed),
    }
    for name, model in models.items():
        results[name] = _fit_eval(name, model, train, (valid, test))

    results["_permutation_control"] = permutation_control(train, valid, num, cat, seed)

    model_dir = settings.paths.models / "baselines"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["logreg"], model_dir / "logreg.joblib")
    results["_meta"] = {
        "train_months": list(settings.modeling.train_months),
        "valid_month": settings.modeling.valid_month,
        "test_month": settings.modeling.test_month,
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "n_features": {"numeric": len(num), "categorical": len(cat)},
    }
    return results


def permutation_control(train, valid, num, cat, seed: int,
                        k: int = PERMUTATIONS, sample: int = PERMUTATION_SAMPLE) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=min(sample, len(train)), replace=False)
    X, y = train.X.iloc[idx], train.y.to_numpy()[idx]
    aucs = []
    for i in range(k):
        model = make_logreg(num, cat, seed).fit(X, np.random.default_rng(seed + i).permutation(y))
        aucs.append(evaluate(valid.y.to_numpy(), model.predict_proba(valid.X)[:, 1])["roc_auc"])
    log.info("[permutation control] valid AUCs %s", [round(a, 4) for a in aucs])
    return {"k": k, "sample": len(idx), "aucs": aucs,
            "mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1))}


def check_permutation_control(results: dict, model: str = "logreg") -> None:
    c = results["_permutation_control"]
    se = c["sd"] / np.sqrt(c["k"])
    real = results[model]["valid"]["roc_auc"]
    if abs(c["mean"] - 0.5) > 3 * se:
        raise AssertionError(f"permuted-label AUCs average {c['mean']:.4f} "
                             f"(> 3 SE = {3 * se:.4f} from 0.5): leakage suspected")
    if real <= max(c["aucs"]):
        raise AssertionError(f"{model} valid AUC {real:.4f} does not beat the best "
                             f"permuted model {max(c['aucs']):.4f}")
    log.info("permutation control passed: mean %.4f +/- %.4f (SE), %s %.4f > max %.4f",
             c["mean"], se, model, real, max(c["aucs"]))


def to_markdown(results: dict) -> str:
    cols = ["roc_auc", "pr_auc", "log_loss", "brier", "ece", "precision_top10", "lift_top10"]
    lines = []
    for split in ("valid", "test"):
        lines += [f"\n**{split}**\n", "| model | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for name, r in results.items():
            if name.startswith("_"):
                continue
            lines.append(f"| {name} | " + " | ".join(f"{r[split][c]:.4f}" for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    settings = get_settings()
    results = run(settings)
    out = settings.paths.reports / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "metrics.md").write_text(to_markdown(results))
    print(to_markdown(results))
    try:
        check_permutation_control(results)
    except AssertionError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
