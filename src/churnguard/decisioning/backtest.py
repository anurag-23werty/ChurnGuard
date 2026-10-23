"""Backtest retention targeting on a held-out month with known outcomes.

Scores the valid month with the *champion* model from the registry (loaded the way
serving loads it), then compares realised net value of contacting the top x% ranked by
expected value, by churn risk and at random, plus sensitivity to the assumed save rate.

Usage:
  python -m churnguard.decisioning.backtest
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import pandas as pd

from churnguard import tracking
from churnguard.config import get_settings
from churnguard.db import connect
from churnguard.decisioning.retention import (
    expected_value,
    monthly_revenue,
    realised_value,
    retained_months,
    select,
    targeting_curve,
)
from churnguard.logging_utils import get_logger, setup_logging
from churnguard.modeling.dataset import load_split

log = get_logger(__name__)

SAVE_RATES = (0.10, 0.20, 0.30)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    args = parser.parse_args(argv)

    settings = get_settings()
    model = tracking.load_champion(settings)
    split = load_split(settings, args.split, connect(settings))
    p = model.predict_proba(split.X)
    revenue = monthly_revenue(split.X["last_amount_paid"], split.X["last_plan_days"])
    y = split.y.to_numpy()
    base = settings.retention

    report = {"split": args.split, "members": len(y), "policy": asdict(base),
              "churn_rate": float(y.mean()), "sensitivity": []}
    for rate in SAVE_RATES:
        policy = replace(base, save_rate=rate)
        ev = expected_value(p, revenue, policy)
        chosen = select(ev, policy.budget)
        report["sensitivity"].append({
            "save_rate": rate,
            "contacted": int(chosen.sum()),
            "contacted_share": float(chosen.mean()),
            "break_even_p_at_149": (policy.incentive_months * 149 + policy.contact_cost)
                                   / (policy.save_rate * 149 * retained_months(policy)),
            "realised_net_value": realised_value(ev, chosen, y, policy),
            "realised_value_contact_everyone": realised_value(
                ev, pd.Series(True, index=ev.index), y, policy),
        })
        log.info("save_rate %.2f: contact %s members (%.1f%%), realised net value %s NTD",
                 rate, f"{int(chosen.sum()):,}", 100 * chosen.mean(),
                 f"{report['sensitivity'][-1]['realised_net_value']:,.0f}")

    ev = expected_value(p, revenue, base)
    curve = targeting_curve(ev, y, base)
    report["targeting_curve"] = curve.to_dict(orient="records")
    out = settings.paths.reports / "decisioning"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"backtest_{args.split}.json").write_text(json.dumps(report, indent=2))
    print(curve.round(0).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
