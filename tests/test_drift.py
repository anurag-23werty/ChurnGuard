from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.monitoring.drift import feature_drift, psi_categorical, psi_numeric, severity

rng = np.random.default_rng(0)


def test_identical_distributions_are_stable():
    x = pd.Series(rng.normal(size=50_000))
    assert psi_numeric(x, x.sample(frac=1, random_state=1)) < 0.001
    assert psi_numeric(x, pd.Series(rng.normal(size=50_000))) < 0.01


def test_shift_is_detected_and_graded():
    ref = pd.Series(rng.normal(size=50_000))
    assert severity(psi_numeric(ref, ref + 0.2)) in {"stable", "moderate"}
    assert severity(psi_numeric(ref, ref + 1.0)) == "major"


def test_psi_matches_hand_computation():
    # Two equal-mass reference bins split at 0; current puts 80% above 0.
    ref = pd.Series([-1.0, 1.0] * 500)
    cur = pd.Series([-1.0] * 200 + [1.0] * 800)
    expected = (0.2 - 0.5) * np.log(0.2 / 0.5) + (0.8 - 0.5) * np.log(0.8 / 0.5)
    assert psi_numeric(ref, cur, n_bins=2) == pytest.approx(expected, rel=1e-6)


def test_jump_in_missing_values_is_drift():
    ref = pd.Series(rng.normal(size=10_000))
    cur = ref.copy()
    cur[: 3000] = np.nan
    assert psi_numeric(ref, cur) > 0.25


def test_categorical_including_unseen_levels():
    ref = pd.Series(["a"] * 500 + ["b"] * 500, dtype="string")
    assert psi_categorical(ref, ref) == pytest.approx(0.0, abs=1e-9)
    new = pd.Series(["a"] * 500 + ["zzz"] * 500, dtype="string")  # b vanished, zzz appeared
    assert psi_categorical(ref, new) > 1.0


def test_feature_drift_table():
    ref = pd.DataFrame({"x": rng.normal(size=5000), "c": rng.choice(["a", "b"], 5000)})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 5000), "c": rng.choice(["a", "b"], 5000)})
    t = feature_drift(ref, cur, ["x"], ["c"])
    assert t.feature.tolist() == ["x", "c"]
    assert t.set_index("feature").loc["x", "severity"] == "major"
    assert t.set_index("feature").loc["c", "severity"] == "stable"
