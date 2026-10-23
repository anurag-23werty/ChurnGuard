from __future__ import annotations

import shutil

from churnguard.data.bronze import ingest_table
from churnguard.data.catalog import LABELS, USER_LOGS
from churnguard.data.dq import FAIL, PASS, WARN, _month_range, run_table
from churnguard.db import connect
from tests.conftest import write_csv


def _checks(report: dict) -> dict[str, dict]:
    return {c["check"]: c for c in report["checks"]}


def test_month_range():
    assert _month_range("2016-11", "2017-02") == ["2016-11", "2016-12", "2017-01", "2017-02"]


def test_user_logs_report_flags_known_dirt(settings, raw_user_logs):
    ingest_table(USER_LOGS, settings)
    report = run_table(connect(settings), settings, "user_logs")
    checks = _checks(report)

    assert report["rows"] == sum(raw_user_logs.values())
    assert checks["total_secs_negative"]["status"] == WARN  # the -9.2e15 row
    assert checks["date_invalid"]["status"] == WARN
    assert checks["month_gaps"]["status"] == PASS
    # Fixture volumes shrink 120 -> 90 -> 60 -> 30 -> 40; 2017-02 is < 0.5x trailing median.
    assert checks["monthly_volume"]["status"] == WARN
    # Fixture repeats (msno, date) within a source.
    assert checks["duplicate_msno_date_source"]["value"] > 0
    assert "total_secs" in report["columns"]


def test_month_gap_fails(settings, raw_user_logs):
    ingest_table(USER_LOGS, settings)
    # Drop a middle month to simulate a broken extract.
    shutil.rmtree(settings.paths.bronze / "user_logs" / "month=2016-12")
    report = run_table(connect(settings), settings, "user_logs")
    assert _checks(report)["month_gaps"]["status"] == FAIL
    assert report["status"] == FAIL


def test_labels_report(settings):
    write_csv(settings.paths.raw / "train.csv", "msno,is_churn", ["a=,0", "b=,0", "c=,1"])
    write_csv(settings.paths.raw / "train_v2.csv", "msno,is_churn", ["a=,0", "a=,0"])
    ingest_table(LABELS, settings)
    checks = _checks(run_table(connect(settings), settings, "labels"))
    assert checks["duplicate_msno_label_month"]["value"] == 1
    assert checks["churn_rate_high"]["status"] == PASS  # 1/5 = 0.20, at but not above warn
