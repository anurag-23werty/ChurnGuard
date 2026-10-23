from __future__ import annotations

import json
import shutil
import subprocess

import duckdb
import pandera.errors
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from churnguard.data.bronze import ingest_table, parse_month_window
from churnguard.data.catalog import LABELS, TRANSACTIONS, USER_LOGS
from churnguard.data.sources import seven_zip_binary
from tests.conftest import write_csv


def _count(glob: str) -> int:
    return duckdb.execute(f"select count(*) from read_parquet('{glob}')").fetchone()[0]


def test_parse_month_window():
    assert parse_month_window("2016-10:2017-03") == (201610, 201703)
    assert parse_month_window("2017-01") == (201701, 201701)
    assert parse_month_window(":2017-01") == (None, 201701)
    assert parse_month_window(None) == (None, None)


def test_user_logs_partitioned_and_reconciled(settings, raw_user_logs):
    manifest = ingest_table(USER_LOGS, settings)
    out = settings.paths.bronze / "user_logs"

    months = sorted(p.name for p in out.glob("month=*"))
    assert months == [
        "month=2016-11", "month=2016-12", "month=2017-01", "month=2017-02", "month=2017-03",
        "month=invalid",
    ]
    assert _count(f"{out}/**/*.parquet") == sum(raw_user_logs.values())
    assert manifest["sources"]["v1"]["invalid_dates"] == {"date": 1}
    assert json.loads((out / "_manifest.json").read_text())["table"] == "user_logs"

    schema = pq.read_schema(out / "month=2016-11" / "v1.parquet")
    assert schema.field("date").type == pa.date32()
    assert schema.field("source").type == pa.string()
    # Every row lands in the partition matching its own date.
    bad = duckdb.execute(f"""
        select count(*) from read_parquet('{out}/month=20*/*.parquet', hive_partitioning=true)
        where strftime(date, '%Y-%m') != month
    """).fetchone()[0]
    assert bad == 0


def test_month_window_filters_but_keeps_invalid(settings, raw_user_logs):
    manifest = ingest_table(USER_LOGS, settings, months="2017-01:2017-02")
    out = settings.paths.bronze / "user_logs"
    assert sorted(p.name for p in out.glob("month=*")) == [
        "month=2017-01", "month=2017-02", "month=invalid",
    ]
    v1 = manifest["sources"]["v1"]
    assert v1["rows_read"] == raw_user_logs["v1"]
    assert v1["rows_written"] == 60 + 30 + 1
    assert v1["rows_read"] == v1["rows_written"] + v1["rows_out_of_window"]


def test_rerun_replaces_table(settings, raw_user_logs):
    ingest_table(USER_LOGS, settings)
    ingest_table(USER_LOGS, settings, months="2017-03")
    out = settings.paths.bronze / "user_logs"
    assert sorted(p.name for p in out.glob("month=*")) == ["month=2017-03", "month=invalid"]
    assert not list(settings.paths.bronze.glob(".staging_*"))
    assert not list(settings.paths.bronze.glob(".old_*"))


def test_labels_get_label_month(settings):
    write_csv(settings.paths.raw / "train.csv", "msno,is_churn", ["a=,0", "b=,1"])
    write_csv(settings.paths.raw / "train_v2.csv", "msno,is_churn", ["a=,1", "c=,0", "d=,0"])
    ingest_table(LABELS, settings)
    rows = duckdb.execute(f"""
        select label_month, count(*) from read_parquet('{settings.paths.bronze}/labels/*.parquet')
        group by 1 order by 1
    """).fetchall()
    assert rows == [("2017-02", 2), ("2017-03", 3)]


def test_contract_violation_fails_and_leaves_no_table(settings):
    header = ("msno,payment_method_id,payment_plan_days,plan_list_price,actual_amount_paid,"
              "is_auto_renew,transaction_date,membership_expire_date,is_cancel")
    write_csv(settings.paths.raw / "transactions.csv", header,
              ["a=,41,30,149,149,2,20170101,20170131,0"])  # is_auto_renew must be 0/1
    write_csv(settings.paths.raw / "transactions_v2.csv", header, [])
    with pytest.raises(pandera.errors.SchemaErrors):
        ingest_table(TRANSACTIONS, settings)
    assert not (settings.paths.bronze / "transactions").exists()


@pytest.mark.skipif(shutil.which("7zz") is None and shutil.which("7z") is None,
                    reason="7-Zip not installed")
def test_reads_7z_archive_via_stream(settings, raw_user_logs):
    for name in ("user_logs.csv", "user_logs_v2.csv"):
        csv = settings.paths.raw / name
        subprocess.run([seven_zip_binary(), "a", "-bd", "-bso0", f"{csv}.7z", str(csv)],
                       check=True)
        csv.unlink()
    ingest_table(USER_LOGS, settings)
    assert _count(f"{settings.paths.bronze}/user_logs/**/*.parquet") == sum(raw_user_logs.values())
