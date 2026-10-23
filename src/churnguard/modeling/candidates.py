"""Candidate models behind one interface: fit(X, y) and predict_proba(X) -> 1-D array.

Every candidate can therefore be compared on the same out-of-time months, registered and
served by the same pyfunc wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from churnguard.features.registry import categorical_features, numeric_features, registry
from churnguard.modeling.gbm import ChurnModel
from churnguard.modeling.preprocessing import make_logreg

# Directions every month agrees on (see the days_to_expiry segment analysis): more
# auto-renew never raises risk; a cancellation never lowers it; more paid-up time left
# never raises it.
MONOTONE = {
    "last_is_auto_renew": -1,
    "auto_renew_share_180d": -1,
    "last_is_cancel": 1,
    "n_cancel_30d": 1,
    "days_to_expiry": -1,
}

ROBUST_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 1000,
    "lambda_l2": 10.0,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "cat_smooth": 30.0,
}


@dataclass
class SklearnChurnModel:
    pipeline: Pipeline
    features: list[str]

    def fit(self, X: pd.DataFrame, y) -> SklearnChurnModel:
        self.pipeline.fit(X[self.features], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]


@dataclass
class BlendModel:
    """Weighted mean of member probabilities. Reason codes come from the first member
    that can explain itself (the tree model), an approximation documented as such."""

    models: list
    weights: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.weights:
            self.weights = [1 / len(self.models)] * len(self.models)
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("blend weights must sum to 1")

    @property
    def features(self) -> list[str]:
        return list(dict.fromkeys(f for m in self.models for f in m.features))

    def fit(self, X: pd.DataFrame, y) -> BlendModel:
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.models,
                                                          strict=True))

    def reason_codes(self, X: pd.DataFrame, k: int = 3):
        for m in self.models:
            if hasattr(m, "reason_codes"):
                return m.reason_codes(X, k)
        raise AttributeError("no member model provides reason codes")


@dataclass
class CalibratedModel:
    """Isotonic recalibration of any candidate's probabilities."""

    model: object
    calibrator: IsotonicRegression

    @property
    def features(self) -> list[str]:
        return self.model.features

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.model.predict_proba(X))

    def reason_codes(self, X: pd.DataFrame, k: int = 3):
        return self.model.reason_codes(X, k)


def _features():
    return list(registry()), categorical_features(), numeric_features()


def make_candidates(tuned_params: dict, num_boost_round: int,
                    seed: int = 42) -> dict[str, Callable[[], object]]:
    """Name -> factory returning a fresh, unfitted candidate."""
    feats, cat, num = _features()

    def lgbm():
        return ChurnModel(feats, cat, {**tuned_params, "seed": seed},
                          num_boost_round=num_boost_round)

    def lgbm_robust():
        return ChurnModel(feats, cat, {**ROBUST_PARAMS, "seed": seed},
                          num_boost_round=400, monotone=MONOTONE)

    def logreg():
        return SklearnChurnModel(make_logreg(num, cat, seed), feats)

    return {
        "logreg": logreg,
        "lgbm": lgbm,
        "lgbm_robust": lgbm_robust,
        "blend_lgbm_logreg": lambda: BlendModel([lgbm(), logreg()]),
        "blend_robust_logreg": lambda: BlendModel([lgbm_robust(), logreg()]),
    }
