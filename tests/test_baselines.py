from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.modeling.baselines import PriorModel, RuleModel, check_permutation_control
from churnguard.modeling.preprocessing import make_logreg, signed_log1p


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({
        "last_is_auto_renew": rng.integers(0, 2, n).astype("float32"),
        "last_is_cancel": rng.integers(0, 2, n).astype("float32"),
        "secs": rng.exponential(1000, n).astype("float32"),
        "city": pd.Series(rng.choice(["1", "5", "13"], n)).astype("string"),
    })
    X.loc[::7, "secs"] = np.nan
    logit = -2 - 2 * X.last_is_auto_renew + 1.5 * X.last_is_cancel
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8"))
    return X, y


def test_signed_log1p_handles_negatives():
    assert signed_log1p(np.array([-np.e + 1, 0, np.e - 1])) == pytest.approx([-1, 0, 1])


def test_prior_predicts_base_rate(toy):
    X, y = toy
    assert PriorModel().fit(X, y).predict_proba(X)[:, 1] == pytest.approx(y.mean())


def test_rule_learns_cell_rates_and_falls_back(toy):
    X, y = toy
    m = RuleModel(smoothing=0).fit(X, y)
    cell = (X.last_is_auto_renew == 0) & (X.last_is_cancel == 1)
    assert m.predict_proba(X[cell])[:, 1] == pytest.approx(y[cell].mean())
    unseen = pd.DataFrame({"last_is_auto_renew": [7.0], "last_is_cancel": [np.nan]})
    assert m.predict_proba(unseen)[0, 1] == pytest.approx(y.mean())


def test_logreg_handles_nans_and_unseen_categories(toy):
    X, y = toy
    model = make_logreg(["last_is_auto_renew", "last_is_cancel", "secs"], ["city"]).fit(X, y)
    new = X.head(3).copy()
    new["city"] = pd.Series(["999", pd.NA, "1"], dtype="string")
    new.loc[new.index[0], "secs"] = np.nan
    p = model.predict_proba(new)[:, 1]
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def _results(aucs, real=0.89):
    return {"logreg": {"valid": {"roc_auc": real}},
            "_permutation_control": {"k": len(aucs), "aucs": aucs, "mean": float(np.mean(aucs)),
                                     "sd": float(np.std(aucs, ddof=1))}}


def test_permutation_control_accepts_wide_symmetric_spread():
    # Real KKBox numbers: random projections land on both sides of 0.5.
    check_permutation_control(_results([0.4175, 0.5888, 0.4340, 0.4554, 0.4720, 0.4449]))


def test_permutation_control_flags_consistently_high_aucs():
    with pytest.raises(AssertionError, match="leakage suspected"):
        check_permutation_control(_results([0.61, 0.62, 0.60, 0.615, 0.605]))


def test_permutation_control_requires_real_model_to_win():
    with pytest.raises(AssertionError, match="does not beat"):
        check_permutation_control(_results([0.45, 0.55, 0.50, 0.48, 0.52], real=0.54))


def test_permutation_control_runs_end_to_end(toy):
    from churnguard.modeling.baselines import permutation_control
    from churnguard.modeling.dataset import Split

    X, y = toy
    split = Split("t", X, y, meta=None)
    c = permutation_control(split, split, ["last_is_auto_renew", "last_is_cancel", "secs"],
                            ["city"], seed=0, k=3, sample=2000)
    assert c["k"] == 3 and c["sample"] == 2000 and all(0 <= a <= 1 for a in c["aucs"])


def test_saved_model_loads_in_another_process(toy, tmp_path):
    """Saved pipelines must load in a fresh process (batch scorer, API). The original bug,
    `__main__.signed_log1p` in a model dumped by `python -m ...baselines`, is prevented
    structurally: pipeline code lives in preprocessing.py, which never runs as __main__."""
    import subprocess
    import sys

    import joblib

    X, y = toy
    path = tmp_path / "m.joblib"
    joblib.dump(make_logreg(["secs"], ["city"]).fit(X, y), path)
    code = (f"import joblib, pandas as pd; m = joblib.load({str(path)!r}); "
            "print(m.predict_proba(pd.DataFrame({'secs': [1.0], "
            "'city': pd.Series(['1'], dtype='string')}))[0, 1])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert 0 < float(out.stdout) < 1
