"""IBM Telco sanity check for the shared modelling code.

Telco is a small, well-studied benchmark where logistic regression reliably lands around
0.84 ROC AUC. Running *our* preprocessing + logreg + metrics on it and asserting that range
catches bugs in the modelling code independently of anything KKBox-specific.

Usage:
  python -m churnguard.modeling.telco
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.config import get_settings
from churnguard.data.download import _kaggle_bin
from churnguard.logging_utils import get_logger, setup_logging
from churnguard.modeling.metrics import evaluate
from churnguard.modeling.preprocessing import make_logreg

log = get_logger(__name__)

DATASET = "blastchar/telco-customer-churn"
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
EXPECTED_AUC = (0.82, 0.88)


def fetch(dest: Path) -> Path:
    csv = dest / CSV_NAME
    if not csv.exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run([_kaggle_bin(), "datasets", "download", DATASET, "-p", str(dest),
                        "--unzip", "-q"], check=True)
    return csv


def load(csv: Path) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = pd.read_csv(csv)
    # TotalCharges is blank for brand-new customers (tenure 0): treat as missing.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df.pop("Churn") == "Yes").astype("int8")
    df = df.drop(columns=["customerID"])
    categorical = [c for c in df.columns if c not in NUMERIC]
    df[NUMERIC] = df[NUMERIC].astype("float32")
    df[categorical] = df[categorical].astype("string")
    return df, y, NUMERIC, categorical


def run(csv: Path, seed: int = 42) -> dict:
    X, y, num, cat = load(csv)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=seed)
    model = make_logreg(num, cat, seed).fit(X_tr, y_tr)
    return {"rows": len(X), "churn_rate": float(y.mean()),
            "test": evaluate(y_te.to_numpy(), model.predict_proba(X_te)[:, 1])}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    settings = get_settings()
    csv = fetch(settings.paths.raw.parent / "external" / "telco")
    result = run(csv, settings.modeling.seed)
    out = settings.paths.reports / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    (out / "telco.json").write_text(json.dumps(result, indent=2))

    auc = result["test"]["roc_auc"]
    lo, hi = EXPECTED_AUC
    log.info("Telco logreg: %d rows, churn %.1f%%, test AUC %.4f (expected %.2f-%.2f)",
             result["rows"], 100 * result["churn_rate"], auc, lo, hi)
    if not lo <= auc <= hi:
        log.error("Telco AUC outside the expected range: the shared modelling code is suspect")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
