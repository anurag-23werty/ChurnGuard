"""Pandera contracts for bronze tables.

Bronze keeps source values as-is, so these are *hard* checks only: things that, if
violated, mean we parsed the file wrong or the source format changed (types, keys,
binary flags, non-negative counts). Known-dirty values (bd = -7000, total_secs < 0,
etc.) are profiled by the DQ report and handled in silver, not rejected here.

Integer columns use pandas nullable dtypes (Int16 etc.) so a null never silently turns an
int column into float64 during validation.
"""

from __future__ import annotations

import pandera.pandas as pa

_msno = pa.Column(str, pa.Check.str_length(min_value=1), nullable=False)
_flag = pa.Check.isin([0, 1])
_non_negative = pa.Check.ge(0)
_date = pa.Column("datetime64[ms]", nullable=True, coerce=True)

MEMBERS = pa.DataFrameSchema(
    {
        "msno": _msno,
        "city": pa.Column("Int16", nullable=True),
        "bd": pa.Column("Int32", nullable=True),
        "gender": pa.Column(str, pa.Check.isin(["male", "female"]), nullable=True),
        "registered_via": pa.Column("Int16", nullable=True),
        "registration_init_time": _date,
    },
)

TRANSACTIONS = pa.DataFrameSchema(
    {
        "msno": _msno,
        "payment_method_id": pa.Column("Int16"),
        "payment_plan_days": pa.Column("Int16", _non_negative),
        "plan_list_price": pa.Column("Int32", _non_negative),
        "actual_amount_paid": pa.Column("Int32", _non_negative),
        "is_auto_renew": pa.Column("Int8", _flag),
        "transaction_date": pa.Column("datetime64[ms]", coerce=True),
        "membership_expire_date": _date,
        "is_cancel": pa.Column("Int8", _flag),
    },
)

USER_LOGS = pa.DataFrameSchema(
    {
        "msno": _msno,
        "date": pa.Column("datetime64[ms]", coerce=True),
        **{
            c: pa.Column("Int32", _non_negative)
            for c in ("num_25", "num_50", "num_75", "num_985", "num_100", "num_unq")
        },
        "total_secs": pa.Column("float64"),
    },
)

LABELS = pa.DataFrameSchema({"msno": _msno, "is_churn": pa.Column("Int8", _flag)})

SCORING_MEMBERS = pa.DataFrameSchema({"msno": _msno})

BY_TABLE: dict[str, pa.DataFrameSchema] = {
    "members": MEMBERS,
    "transactions": TRANSACTIONS,
    "user_logs": USER_LOGS,
    "labels": LABELS,
    "scoring_members": SCORING_MEMBERS,
}
