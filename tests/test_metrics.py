from __future__ import annotations

import numpy as np
import pytest

from churnguard.modeling.metrics import evaluate, expected_calibration_error, top_k


def test_perfect_ranking():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    p = np.linspace(0.01, 0.99, 10)
    m = evaluate(y, p)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == pytest.approx(1.0)
    assert m["precision_top10"] == 1.0 and m["lift_top10"] == pytest.approx(5.0)
    assert m["recall_top10"] == 0.5


def test_constant_prediction_has_no_ranking_skill_but_is_calibrated():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.07).astype(int)
    m = evaluate(y, np.full(len(y), y.mean()))
    assert m["roc_auc"] == pytest.approx(0.5)
    # Equal-mass bins of a constant score hold arbitrary rows, so per-bin rates differ by
    # sampling noise only (~sqrt(q(1-q)/2000) = 0.006 per bin).
    assert m["ece"] < 0.01
    # log loss of the base rate = entropy of the label
    q = y.mean()
    assert m["log_loss"] == pytest.approx(-(q * np.log(q) + (1 - q) * np.log(1 - q)))


def test_ece_detects_miscalibration():
    rng = np.random.default_rng(1)
    p = rng.random(50_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 1.5, 0, 1)) > 0.1


def test_extreme_probabilities_are_clipped():
    m = evaluate([0, 1], [1.0, 0.0])  # confidently wrong
    assert np.isfinite(m["log_loss"]) and m["log_loss"] > 30


def test_top_k_rounds_to_at_least_one():
    assert top_k(np.array([1, 0, 0]), np.array([0.9, 0.1, 0.2]), 0.01)["precision"] == 1.0
