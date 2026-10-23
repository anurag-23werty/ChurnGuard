from __future__ import annotations

import json

import duckdb
import pytest

from churnguard.data.bronze import ingest_table
from churnguard.data.catalog import TABLES
from churnguard.data.silver import ContractError, build_table
from churnguard.db import connect
from tests.conftest import write_csv

TXN_HEADER = ("msno,payment_method_id,payment_plan_days,plan_list_price,actual_amount_paid,"
              "is_auto_renew,transaction_date,membership_expire_date,is_cancel")
LOG_HEADER = "msno,date,num_25,num_50,num_75,num_985,num_100,num_unq,total_secs"


@pytest.fixture
def bronze(settings):
    raw = settings.paths.raw
    write_csv(raw / "members_v3.csv", "msno,city,bd,gender,registered_via,registration_init_time", [
        "a=,1,0,,7,20150101",        # bd=0 -> unknown age, missing gender
        "b=,5,28,female,9,20160301",
        "c=,13,-7000,male,4,20140505",
    ])
    write_csv(raw / "transactions.csv", TXN_HEADER, [
        "a=,41,30,149,149,1,20170101,20170131,0",
        "a=,41,30,149,149,1,20170101,20170131,0",   # exact duplicate -> dropped
        "a=,41,30,149,149,1,20170101,20170105,1",   # same day, different content -> kept
        "b=,39,30,149,149,1,20170110,19700101,0",   # placeholder expiry -> null
    ])
    write_csv(raw / "transactions_v2.csv", TXN_HEADER, [
        "a=,41,30,149,149,1,20170101,20170131,0",   # also in v1 -> one row, source v2
        "d=,41,30,149,149,1,20160615,20160715,0",   # historical row only in v2 -> kept
        "b=,39,30,149,149,1,20170310,20170410,0",
    ])
    write_csv(raw / "user_logs.csv", LOG_HEADER, [
        "a=,20170201,1,0,0,0,5,4,1234.5",
        "b=,20170201,1,0,0,0,5,4,-9.2e15",           # negative secs -> clipped to 0
        "a=,20170202,1,0,0,0,5,4,100000",            # > 24h -> clipped to 86400
    ])
    write_csv(raw / "user_logs_v2.csv", LOG_HEADER, ["c=,20170301,0,1,0,0,3,2,600"])
    write_csv(raw / "train.csv", "msno,is_churn", ["a=,0", "b=,1"])
    write_csv(raw / "train_v2.csv", "msno,is_churn", ["a=,1", "b=,0", "e=,0"])
    write_csv(raw / "sample_submission_v2.csv", "msno,is_churn", ["a=,0", "c=,0"])
    for spec in TABLES.values():
        ingest_table(spec, settings)
    return settings


def _rows(sql: str) -> list[tuple]:
    return duckdb.execute(sql).fetchall()


def test_member_ids_cover_all_tables_and_are_append_only(bronze):
    con = connect(bronze)
    report = build_table(con, bronze, "member_ids")
    path = bronze.paths.silver / "member_ids" / "member_ids.parquet"
    first = dict(_rows(f"SELECT msno, member_id FROM '{path}'"))
    assert set(first) == {"a=", "b=", "c=", "d=", "e="}
    assert report["new_ids"] == 5

    # A new member appears later: existing ids must not move.
    with open(bronze.paths.raw / "train_v2.csv", "a") as f:
        f.write("aa=,1\n")
    ingest_table(TABLES["labels"], bronze)
    report = build_table(con, bronze, "member_ids")
    second = dict(_rows(f"SELECT msno, member_id FROM '{path}'"))
    assert report["new_ids"] == 1
    assert {k: second[k] for k in first} == first
    assert second["aa="] == max(first.values()) + 1


def test_members_cleaning(bronze):
    con = connect(bronze)
    build_table(con, bronze, "member_ids")
    report = build_table(con, bronze, "members")
    rows = _rows(f"""
        SELECT i.msno, m.age, m.gender FROM '{bronze.paths.silver}/members/members.parquet' m
        JOIN '{bronze.paths.silver}/member_ids/member_ids.parquet' i USING (member_id)
        ORDER BY 1""")
    assert rows == [("a=", None, "unknown"), ("b=", 28, "female"), ("c=", None, "male")]
    assert report["rules"]["age_nulled_out_of_range"] == 2


def test_transactions_dedup_and_union(bronze):
    con = connect(bronze)
    build_table(con, bronze, "member_ids")
    report = build_table(con, bronze, "transactions")
    t = f"'{bronze.paths.silver}/transactions/transactions.parquet'"
    assert report["rows_bronze"] == 7
    assert report["rows_silver"] == 5  # 2 exact dups (within v1, and v1 vs v2) removed
    assert report["rules"]["exact_duplicates_dropped"] == 2
    assert report["rules"]["expire_date_placeholder_nulled"] == 1
    # The v1/v2 tie keeps the v2 tag; same-day different-content row survives.
    assert _rows(f"""
        SELECT source, is_cancel FROM {t}
        WHERE transaction_date = DATE '2017-01-01' ORDER BY is_cancel""") == [
        ("v2", False), ("v1", True)]
    assert _rows(f"SELECT count(*) FROM {t} WHERE membership_expire_date IS NULL") == [(1,)]
    assert _rows(f"SELECT count(*) FROM {t} WHERE transaction_date < DATE '2017-01-01'") == [(1,)]


def test_user_logs_clipped_sorted_partitioned(bronze):
    con = connect(bronze)
    build_table(con, bronze, "member_ids")
    report = build_table(con, bronze, "user_logs")
    out = bronze.paths.silver / "user_logs"
    assert sorted(p.name for p in out.glob("month=*")) == ["month=2017-02", "month=2017-03"]
    assert report["rows_bronze"] == report["rows_silver"] == 4
    assert report["rules"]["total_secs_clipped"] == 2
    secs = sorted(r[0] for r in _rows(f"SELECT total_secs FROM '{out}/**/*.parquet'"))
    assert secs == [0, 600, 1234.5, 86400]
    schema = dict(_rows(f"""
        SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM '{out}/**/*.parquet')"""))
    assert "msno" not in schema and schema["member_id"] == "INTEGER"
    assert json.loads((out / "_manifest.json").read_text())["months"]["2017-02"]["rows_silver"] == 3


def test_labels_and_scoring(bronze):
    con = connect(bronze)
    build_table(con, bronze, "member_ids")
    report = build_table(con, bronze, "labels")
    assert report["rows_silver"] == 5
    assert report["churn_rate"] == {"2017-02": 0.5, "2017-03": round(1 / 3, 5)}
    assert build_table(con, bronze, "scoring_members")["rows_silver"] == 2


def test_contract_violation_keeps_previous_table(bronze):
    con = connect(bronze)
    build_table(con, bronze, "member_ids")
    build_table(con, bronze, "labels")
    before = (bronze.paths.silver / "labels" / "labels.parquet").read_bytes()

    # Conflicting labels for the same member-month must fail the build.
    with open(bronze.paths.raw / "train_v2.csv", "a") as f:
        f.write("a=,0\n")
    ingest_table(TABLES["labels"], bronze)
    with pytest.raises(ContractError, match="duplicate_member_month"):
        build_table(con, bronze, "labels")
    assert (bronze.paths.silver / "labels" / "labels.parquet").read_bytes() == before
