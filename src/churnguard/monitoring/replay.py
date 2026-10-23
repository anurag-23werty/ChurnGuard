"""Replay monitoring over history: would it have flagged the bad months?

For each month M, drift is measured against the two months before it, as a monthly job
running at the time would have. Performance is skipped: the champion was trained on most of
these months, so its metrics there are in-sample.

Usage:
  python -m churnguard.monitoring.replay --months 2016-12:2017-04
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from churnguard import tracking
from churnguard.config import get_settings
from churnguard.db import connect
from churnguard.labels.spine import months_between
from churnguard.logging_utils import get_logger, setup_logging
from churnguard.monitoring.drift import MAJOR, MODERATE
from churnguard.monitoring.monitor import _prev_months, monitor_month

log = get_logger(__name__)


def replay(settings, months: list[str], model, window: int = 2) -> pd.DataFrame:
    con = connect(settings)
    rows = []
    for month in months:
        ref = sorted(_prev_months(month, window))
        r = monitor_month(settings, month, model, ref, con, check_performance=False)
        drift = pd.DataFrame(r["feature_drift"])
        drift = drift[~drift.expected_drift]  # reported in the monthly report, not counted
        rows.append({
            "month": month, "reference": f"{ref[0]}..{ref[-1]}",
            "score_psi": r["score_drift"]["psi"],
            "mean_p": round(r["score_drift"]["mean_p_current"], 4),
            "major": int((drift.psi > MAJOR).sum()),
            "moderate": int(((drift.psi > MODERATE) & (drift.psi <= MAJOR)).sum()),
            "top_drift": ", ".join(f"{f} {p:.2f}" for f, p in
                                   drift.head(3)[["feature", "psi"]].itertuples(index=False)),
            "decision": r["decision"]["status"],
        })
        log.info("%s: score PSI %.4f, %d major / %d moderate -> %s", month, rows[-1]["score_psi"],
                 rows[-1]["major"], rows[-1]["moderate"], rows[-1]["decision"])
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--months", default="2016-12:2017-04")
    args = parser.parse_args(argv)
    settings = get_settings()
    table = replay(settings, months_between(args.months), tracking.load_champion(settings))
    out = settings.paths.reports / "monitoring"
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(json.dumps(table.to_dict(orient="records"), indent=2))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
