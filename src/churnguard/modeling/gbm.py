"""LightGBM churn model: everything needed to score a member in one picklable object.

ChurnModel holds the booster, the feature order, the categorical levels seen in training
(so serving encodes "city 13" exactly as training did; unseen levels become NaN, which
LightGBM routes like missing), an optional isotonic calibrator, and SHAP reason codes via
LightGBM's exact TreeSHAP (`pred_contrib`), so no separate `shap` dependency.

Lives in an importable module (never __main__) so pickles reference it by module path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "num_threads": 6,
    "seed": 42,
    "deterministic": True,
    "force_col_wise": True,
}


@dataclass
class ChurnModel:
    features: list[str]
    categorical: list[str]
    params: dict = field(default_factory=dict)
    booster: lgb.Booster | None = None
    levels: dict[str, list[str]] = field(default_factory=dict)
    calibrator: IsotonicRegression | None = None
    best_iteration: int | None = None
    num_boost_round: int = 200  # used when fit() is called without early stopping
    # Monotone constraints by feature name: +1 = can only raise churn risk, -1 = only lower it.
    monotone: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ encoding
    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features].copy()
        for c in self.categorical:
            # Unseen levels -> missing explicitly (pandas will soon raise instead).
            values = X[c].astype("string")
            values = values.where(values.isin(self.levels[c]))
            X[c] = pd.Categorical(values, categories=self.levels[c])
        return X

    def _learn_levels(self, X: pd.DataFrame) -> None:
        self.levels = {c: sorted(X[c].dropna().astype(str).unique().tolist())
                       for c in self.categorical}

    # ------------------------------------------------------------------ training
    def _lgb_params(self) -> dict:
        params = {**BASE_PARAMS, **self.params}
        if self.monotone:
            unknown = set(self.monotone) - set(self.features)
            if unknown:
                raise ValueError(f"monotone constraints on unknown features: {sorted(unknown)}")
            params["monotone_constraints"] = [self.monotone.get(f, 0) for f in self.features]
            params["monotone_constraints_method"] = "advanced"
        return params

    def fit(self, X: pd.DataFrame, y, X_valid: pd.DataFrame | None = None, y_valid=None,
            num_boost_round: int | None = None,
            early_stopping_rounds: int = 100) -> ChurnModel:
        self._learn_levels(X)
        params = self._lgb_params()
        if num_boost_round is None:
            num_boost_round = 2000 if X_valid is not None else self.num_boost_round
        train = lgb.Dataset(self._encode(X), label=np.asarray(y),
                            categorical_feature=self.categorical, free_raw_data=True)
        callbacks, valid_sets = [], []
        if X_valid is not None:
            valid_sets = [lgb.Dataset(self._encode(X_valid), label=np.asarray(y_valid),
                                      reference=train)]
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        self.booster = lgb.train(params, train, num_boost_round=num_boost_round,
                                 valid_sets=valid_sets, callbacks=callbacks)
        self.best_iteration = self.booster.best_iteration or self.booster.current_iteration()
        self.calibrator = None  # a refit invalidates any previous calibration
        return self

    # ------------------------------------------------------------------ scoring
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._encode(X), num_iteration=self.best_iteration)

    def calibrate(self, raw: np.ndarray, y) -> ChurnModel:
        """Fit isotonic on *out-of-time* raw scores (never on the rows it is evaluated on)."""
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        self.calibrator.fit(raw, np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    # ------------------------------------------------------------------ explanation
    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions in log-odds of the *uncalibrated* model; the extra
        `_base` column is the expected value. Rows sum to the raw logit."""
        contrib = self.booster.predict(self._encode(X), num_iteration=self.best_iteration,
                                       pred_contrib=True)
        return pd.DataFrame(contrib, columns=[*self.features, "_base"], index=X.index)

    def reason_codes(self, X: pd.DataFrame, k: int = 3) -> list[list[tuple[str, float]]]:
        """Top-k features pushing each member *towards* churn, largest first."""
        sv = self.shap_values(X).drop(columns="_base").to_numpy()
        order = np.argsort(-sv, axis=1)[:, :k]
        return [[(self.features[j], float(sv[i, j])) for j in row if sv[i, j] > 0]
                for i, row in enumerate(order)]

    def feature_importance(self, X: pd.DataFrame) -> pd.Series:
        """Global importance = mean |SHAP| over X."""
        sv = self.shap_values(X).drop(columns="_base")
        return sv.abs().mean().sort_values(ascending=False)
