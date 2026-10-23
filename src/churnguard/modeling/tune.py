"""Rolling-origin cross-validation and Optuna tuning, entirely inside the training months.

Folds expand forward in time (train <= M-1, validate on M), the way the model is used:
fit on history, score next month. Valid (Feb) and test (Mar) are never touched here: early
stopping, the number of boosting rounds and the calibrator all come from these folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from churnguard.logging_utils import get_logger
from churnguard.modeling.gbm import ChurnModel
from churnguard.modeling.metrics import evaluate

log = get_logger(__name__)

ROUNDS_SAFETY = 1.1  # final model sees more data than any fold: train a bit longer


@dataclass(frozen=True)
class Fold:
    train_months: tuple[str, ...]
    valid_month: str


@dataclass
class CVResult:
    log_loss: float
    fold_log_loss: list[float]
    best_iterations: list[int]
    oof_raw: np.ndarray  # out-of-time raw scores for every fold's valid month
    oof_y: np.ndarray
    oof_month: np.ndarray

    @property
    def num_boost_round(self) -> int:
        return int(np.ceil(np.mean(self.best_iterations) * ROUNDS_SAFETY))


def rolling_folds(months: tuple[str, ...] | list[str], min_train: int = 1) -> list[Fold]:
    months = sorted(months)
    return [Fold(tuple(months[:i]), months[i]) for i in range(min_train, len(months))]


def cross_validate(X: pd.DataFrame, y: pd.Series, month: pd.Series, folds: list[Fold],
                   features: list[str], categorical: list[str], params: dict,
                   trial: optuna.Trial | None = None) -> CVResult:
    losses, iters, raws, ys, ms = [], [], [], [], []
    month = month.to_numpy()
    for i, fold in enumerate(folds):
        tr, va = np.isin(month, fold.train_months), month == fold.valid_month
        model = ChurnModel(features, categorical, params).fit(X[tr], y[tr], X[va], y[va])
        raw = model.predict_raw(X[va])
        loss = evaluate(y[va].to_numpy(), raw)["log_loss"]
        losses.append(loss)
        iters.append(model.best_iteration)
        raws.append(raw), ys.append(y[va].to_numpy()), ms.append(month[va])
        if trial is not None:
            trial.report(float(np.mean(losses)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return CVResult(float(np.mean(losses)), losses, iters,
                    np.concatenate(raws), np.concatenate(ys), np.concatenate(ms))


def search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 50.0, log=True),
    }


def tune(X, y, month, folds, features, categorical, n_trials: int, seed: int,
         timeout_s: int | None = None) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        res = cross_validate(X, y, month, folds, features, categorical, params, trial)
        trial.set_user_attr("best_iterations", res.best_iterations)
        log.info("[trial %d] cv log loss %.5f %s", trial.number, res.log_loss,
                 {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()})
        return res.log_loss

    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, gc_after_trial=True)
    return study
