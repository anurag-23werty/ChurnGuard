"""Population stability index (PSI) for features and scores.

PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%).  Conventional reading:
< 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.

Numeric features use reference quantile bins (so each holds ~equal reference mass) plus a
separate bin for missing values: a jump in nulls is drift too. Categorical features compare
level shares; levels unseen in the reference are pooled into one bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10
EPS = 1e-4  # floor on bin shares so empty bins do not give log(0)
MODERATE, MAJOR = 0.10, 0.25


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    r, c = np.clip(ref_share, EPS, None), np.clip(cur_share, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref: pd.Series, cur: pd.Series, n_bins: int = N_BINS) -> float:
    ref, cur = pd.to_numeric(ref).astype(float), pd.to_numeric(cur).astype(float)
    ref_valid = ref.dropna().to_numpy()
    if len(ref_valid) == 0:
        edges = np.array([])
    else:
        edges = np.unique(np.quantile(ref_valid, np.linspace(0, 1, n_bins + 1)[1:-1]))

    def shares(x: pd.Series) -> np.ndarray:
        valid = x.dropna().to_numpy()
        counts = np.bincount(np.searchsorted(edges, valid, side="right"),
                             minlength=len(edges) + 1)
        return np.append(counts, x.isna().sum()) / max(len(x), 1)

    return _psi(shares(ref), shares(cur))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref = ref.astype("string").fillna("<missing>")
    cur = cur.astype("string").fillna("<missing>")
    levels = ref.value_counts(normalize=True)
    cur_counts = cur.value_counts(normalize=True)
    unseen = cur_counts[~cur_counts.index.isin(levels.index)].sum()
    r = np.append(levels.to_numpy(), 0.0)
    c = np.append(cur_counts.reindex(levels.index, fill_value=0).to_numpy(), unseen)
    return _psi(r, c)


def severity(psi: float) -> str:
    return "major" if psi > MAJOR else "moderate" if psi > MODERATE else "stable"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, numeric: list[str],
                  categorical: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "kind": "numeric", "psi": psi_numeric(ref[f], cur[f]),
             "ref_null": float(ref[f].isna().mean()), "cur_null": float(cur[f].isna().mean())}
            for f in numeric]
    rows += [{"feature": f, "kind": "categorical", "psi": psi_categorical(ref[f], cur[f]),
              "ref_null": float(ref[f].isna().mean()), "cur_null": float(cur[f].isna().mean())}
             for f in categorical]
    out = pd.DataFrame(rows)
    out["severity"] = out.psi.map(severity)
    return out.sort_values("psi", ascending=False, ignore_index=True)
