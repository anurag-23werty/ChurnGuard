"""Shared preprocessing and model factories.

Kept out of any module that runs as `__main__`: pickled pipelines reference these
functions by module path, so a model saved by `python -m churnguard.modeling.baselines`
must point at `churnguard.modeling.preprocessing.signed_log1p`, not `__main__.signed_log1p`,
or no other process (batch scorer, API) can load it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def as_category_str(X):
    """Codes as plain str with an explicit 'missing' level. Handles pd.NA (pandas string
    dtype), None and NaN alike, which SimpleImputer does not."""
    X = pd.DataFrame(X)
    return X.astype(object).where(X.notna(), "missing").astype(str)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        FunctionTransformer(signed_log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    cat = make_pipeline(
        FunctionTransformer(as_category_str, feature_names_out="one-to-one"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001,
                      sparse_output=False),
    )
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def make_logreg(numeric: list[str], categorical: list[str], seed: int = 42) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(numeric, categorical)),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
    ])
