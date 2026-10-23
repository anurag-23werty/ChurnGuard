"""Monthly monitoring: data quality, feature + score drift, matured performance, decision.

  ok       nothing material
  warn     moderate score drift, any major feature drift, odd volume or null jumps
  retrain  major score drift, >= N features with major drift, or a *complete* matured
           month with AUC well below the champion's out-of-time AUC or ECE too high

Usage:
  python -m churnguard.monitoring.monitor                 # latest scored month
  python -m churnguard.monitoring.monitor --month 2017-04
Exit code 3 = retrain recommended (for schedulers), 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys

import duckdb
import numpy as np
import pandas as pd

from churnguard import tracking
from churnguard.config import MonitoringConfig, Settings, get_settings
from churnguard.db import connect
from churnguard.features.registry import (
    categorical_features,
    expected_drift_features,
    numeric_features,
)
from churnguard.labels.spine import months_between
from churnguard.logging_utils import get_logger, setup_logging
from churnguard.modeling.dataset import prepare_features
from churnguard.monitoring.drift import MAJOR, feature_drift, psi_numeric, severity
from churnguard.monitoring.performance import month_features, performance, scores_dir

log = get_logger(__name__)
RETRAIN_EXIT = 3


def reference_features(settings: Settings, months, n: int, seed: int) -> pd.DataFrame:
    frames = [month_features(settings, m) for m in months]
    ref = pd.concat([f for f in frames if f is not None], ignore_index=True)
    return ref.sample(min(n, len(ref)), random_state=seed, ignore_index=True)


def _prev_months(month: str, k: int) -> list[str]:
    y, m = map(int, month.split("-"))
    out = []
    for _ in range(k):
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        out.append(f"{y:04d}-{m:02d}")
    return out


def data_quality(ref: pd.DataFrame, cur: pd.DataFrame, p_cur: np.ndarray,
                 ref_rows_per_month: float, cfg: MonitoringConfig) -> dict:
    ratio = len(cur) / ref_rows_per_month if ref_rows_per_month else float("nan")
    num = numeric_features()
    jumps = (cur[num].isna().mean() - ref[num].isna().mean()).abs().sort_values(ascending=False)
    return {
        "rows": len(cur), "volume_ratio": round(ratio, 4),
        "volume_ok": bool(cfg.volume_ratio[0] <= ratio <= cfg.volume_ratio[1]),
        "max_null_jump": {"feature": jumps.index[0], "delta": round(float(jumps.iloc[0]), 4)},
        "p_in_range": bool(((p_cur >= 0) & (p_cur <= 1)).all()),
    }


def decide(dq: dict, score_psi: float, drift: pd.DataFrame, perf: list[dict],
           expected_auc: float | None, cfg: MonitoringConfig,
           expected_drift: set[str] = frozenset()) -> dict:
    """Months the champion was trained on (`in_sample`) and immature months never count;
    nor do features that drift by construction (registry `expected_drift`)."""
    retrain, warn = [], []
    drift = drift[~drift.feature.isin(expected_drift)]
    n_major = int((drift.psi > MAJOR).sum())
    if score_psi > cfg.score_psi_retrain:
        retrain.append(f"score PSI {score_psi:.3f} > {cfg.score_psi_retrain}")
    elif score_psi > cfg.score_psi_warn:
        warn.append(f"score PSI {score_psi:.3f} > {cfg.score_psi_warn}")
    if n_major >= cfg.major_features_retrain:
        retrain.append(f"{n_major} features with major drift (>= {cfg.major_features_retrain})")
    elif n_major:
        warn.append(f"major drift in {', '.join(drift.feature[drift.psi > MAJOR])}")
    for p in perf:
        m = p.get("metrics")
        if p.get("status") != "complete" or not m or p.get("in_sample"):
            continue
        if expected_auc is not None and m["roc_auc"] < expected_auc - cfg.auc_drop_retrain:
            retrain.append(f"{p['month']} AUC {m['roc_auc']:.4f} < expected "
                           f"{expected_auc:.4f} - {cfg.auc_drop_retrain}")
        if m["ece"] > cfg.ece_retrain:
            retrain.append(f"{p['month']} ECE {m['ece']:.4f} > {cfg.ece_retrain}")
    if not dq["volume_ok"]:
        warn.append(f"volume ratio {dq['volume_ratio']} outside {list(cfg.volume_ratio)}")
    if dq["max_null_jump"]["delta"] > cfg.null_jump_warn:
        warn.append(f"null rate of {dq['max_null_jump']['feature']} moved "
                    f"{dq['max_null_jump']['delta']:.3f}")
    if not dq["p_in_range"]:
        retrain.append("probabilities outside [0, 1]")
    status = "retrain" if retrain else "warn" if warn else "ok"
    return {"status": status, "retrain_reasons": retrain, "warnings": warn}


def monitor_month(settings: Settings, month: str, model, reference_months: list[str],
                  con: duckdb.DuckDBPyConnection, expected_auc: float | None = None,
                  check_performance: bool = True, last_training_month: str | None = None
                  ) -> dict:
    cfg = settings.monitoring
    cur = month_features(settings, month)
    if cur is None:
        raise FileNotFoundError(f"no features for {month}: score it first (make score)")
    ref = reference_features(settings, reference_months, cfg.reference_sample,
                             settings.modeling.seed)
    ref_rows = np.mean([len(f) for m in reference_months
                        if (f := month_features(settings, m)) is not None])

    served = scores_dir(settings, month) / "data.parquet"
    p_cur = (pd.read_parquet(served, columns=["p_churn"]).p_churn.to_numpy()
             if served.exists() else model.predict_proba(prepare_features(cur)))
    p_ref = model.predict_proba(prepare_features(ref))
    score_psi = psi_numeric(pd.Series(p_ref), pd.Series(p_cur))

    # Drift on what the model sees: same dtypes and category codes as its inputs.
    drift = feature_drift(prepare_features(ref), prepare_features(cur), numeric_features(),
                          categorical_features())
    drift["expected_drift"] = drift.feature.isin(expected_drift_features())
    dq = data_quality(ref, cur, p_cur, ref_rows, cfg)
    last_training_month = last_training_month or reference_months[-1]
    perf = ([{**performance(settings, m, con, model), "in_sample": m <= last_training_month}
             for m in [month, *_prev_months(month, 2)]] if check_performance else [])
    return {
        "month": month, "reference_months": reference_months,
        "data_quality": dq,
        "score_drift": {"psi": round(score_psi, 4), "severity": severity(score_psi),
                        "mean_p_reference": float(p_ref.mean()),
                        "mean_p_current": float(p_cur.mean())},
        "feature_drift": drift.round(4).to_dict(orient="records"),
        "performance": perf,
        "decision": decide(dq, score_psi, drift, perf, expected_auc, cfg,
                           expected_drift_features()),
    }


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", help="YYYY-MM (default: latest scored month)")
    parser.add_argument("--reference", help="YYYY-MM:YYYY-MM (default: training months)")
    args = parser.parse_args(argv)

    scored = sorted(p.name.removeprefix("score_month=")
                    for p in (settings.paths.gold / "scores").glob("score_month=*"))
    month = args.month or (scored[-1] if scored else None)
    if month is None:
        log.error("nothing scored yet: run `make score` first")
        return 1
    # Default reference = what the champion was fit on (train months + valid month).
    reference = (months_between(args.reference) if args.reference
                 else [*settings.modeling.train_months, settings.modeling.valid_month])
    model = tracking.load_champion(settings)
    expected = (tracking.champion_metrics(settings) or {}).get("mean_roc_auc")
    report = monitor_month(settings, month, model, reference, connect(settings), expected)

    out = settings.paths.reports / "monitoring"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"month={month}.json").write_text(json.dumps(report, indent=2, default=str))
    d, s = report["decision"], report["score_drift"]
    log.info("%s vs %s: score PSI %.4f (%s), mean p %.4f -> %.4f", month,
             f"{reference[0]}..{reference[-1]}", s["psi"], s["severity"],
             s["mean_p_reference"], s["mean_p_current"])
    for f in report["feature_drift"][:5]:
        log.info("  %-26s PSI %.4f %s", f["feature"], f["psi"], f["severity"])
    for p in report["performance"]:
        log.info("  performance %s: %s (%s decided)%s%s", p["month"], p["status"],
                 p.get("decided_share"), f" AUC {p['metrics']['roc_auc']:.4f}"
                 if "metrics" in p else "", " [in-sample, ignored]" if p["in_sample"] else "")
    log.info("decision: %s %s", d["status"].upper(), d["retrain_reasons"] or d["warnings"])
    return RETRAIN_EXIT if d["status"] == "retrain" else 0


if __name__ == "__main__":
    sys.exit(main())
