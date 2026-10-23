"""Data contract for the KKBox raw files.

One entry per logical table. Each table can come from several raw files (v1 + v2 refresh);
`source` is recorded on every bronze row so silver can de-duplicate the overlap explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa


@dataclass(frozen=True)
class SourceFile:
    archive: str  # file name as published on Kaggle
    source: str  # short tag written to the bronze `source` column


@dataclass(frozen=True)
class TableSpec:
    name: str
    files: tuple[SourceFile, ...]
    # Types used to read the CSV. YYYYMMDD ints are read as int32 and converted to dates.
    read_schema: pa.Schema
    date_columns: tuple[str, ...] = ()
    # Date column that defines the month partition; None = single unpartitioned file.
    partition_by: str | None = None
    # Columns that must never be null in bronze.
    required: tuple[str, ...] = ("msno",)
    # Constant columns added per source (e.g. label month for train vs train_v2).
    extra_by_source: dict[str, dict[str, str]] = field(default_factory=dict)


MEMBERS = TableSpec(
    name="members",
    files=(SourceFile("members_v3.csv.7z", "v3"),),
    read_schema=pa.schema(
        [
            ("msno", pa.string()),
            ("city", pa.int16()),
            ("bd", pa.int32()),
            ("gender", pa.string()),
            ("registered_via", pa.int16()),
            ("registration_init_time", pa.int32()),
        ]
    ),
    date_columns=("registration_init_time",),
)

TRANSACTIONS = TableSpec(
    name="transactions",
    files=(
        SourceFile("transactions.csv.7z", "v1"),
        SourceFile("transactions_v2.csv.7z", "v2"),
    ),
    read_schema=pa.schema(
        [
            ("msno", pa.string()),
            ("payment_method_id", pa.int16()),
            ("payment_plan_days", pa.int16()),
            ("plan_list_price", pa.int32()),
            ("actual_amount_paid", pa.int32()),
            ("is_auto_renew", pa.int8()),
            ("transaction_date", pa.int32()),
            ("membership_expire_date", pa.int32()),
            ("is_cancel", pa.int8()),
        ]
    ),
    date_columns=("transaction_date", "membership_expire_date"),
    partition_by="transaction_date",
    required=("msno", "transaction_date", "membership_expire_date"),
)

USER_LOGS = TableSpec(
    name="user_logs",
    files=(
        SourceFile("user_logs.csv.7z", "v1"),
        SourceFile("user_logs_v2.csv.7z", "v2"),
    ),
    read_schema=pa.schema(
        [
            ("msno", pa.string()),
            ("date", pa.int32()),
            ("num_25", pa.int32()),
            ("num_50", pa.int32()),
            ("num_75", pa.int32()),
            ("num_985", pa.int32()),
            ("num_100", pa.int32()),
            ("num_unq", pa.int32()),
            ("total_secs", pa.float64()),
        ]
    ),
    date_columns=("date",),
    partition_by="date",
    required=("msno", "date"),
)

# Official labels: train = expiries in Feb 2017, train_v2 = expiries in Mar 2017.
LABELS = TableSpec(
    name="labels",
    files=(
        SourceFile("train.csv.7z", "train"),
        SourceFile("train_v2.csv.7z", "train_v2"),
    ),
    read_schema=pa.schema([("msno", pa.string()), ("is_churn", pa.int8())]),
    required=("msno", "is_churn"),
    extra_by_source={
        "train": {"label_month": "2017-02"},
        "train_v2": {"label_month": "2017-03"},
    },
)

# Kaggle test population (expiries in Apr 2017, labels hidden). Used later as a scoring cohort.
SCORING_MEMBERS = TableSpec(
    name="scoring_members",
    files=(SourceFile("sample_submission_v2.csv.7z", "v2"),),
    read_schema=pa.schema([("msno", pa.string()), ("is_churn", pa.int8())]),
)

TABLES: dict[str, TableSpec] = {
    t.name: t for t in (MEMBERS, TRANSACTIONS, USER_LOGS, LABELS, SCORING_MEMBERS)
}
