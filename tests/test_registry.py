from __future__ import annotations

import pytest

from churnguard.features.registry import (
    RegistryError,
    by_family,
    check_columns,
    load_registry,
    registry,
)


def test_windows_expand_and_descriptions_fill():
    reg = registry()
    assert {"n_txn_30d", "n_txn_90d", "n_txn_180d", "n_txn_365d"} <= set(reg)
    assert reg["total_secs_7d"].window_days == 7
    assert reg["total_secs_7d"].description == "Seconds listened in the last 7 days"
    assert reg["tenure_days"].window_days is None


def test_engagement_windows_fit_log_coverage():
    # silver/user_logs starts 2016-07-01; earliest as_of_date is 2016-10-01 (92 days).
    assert max(f.window_days or 0 for f in by_family("engagement")) <= 90


def test_check_columns_reports_missing_and_unregistered():
    names = [f.name for f in by_family("member")]
    check_columns(names, "member")
    with pytest.raises(RegistryError, match=r"missing=\['age'\] unregistered=\['shoe_size'\]"):
        check_columns([n for n in names if n != "age"] + ["shoe_size"], "member")


def test_duplicate_feature_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("""
families:
  a:
    source: t
    features:
      - {name: x, dtype: int, windows: [7], description: d}
      - {name: x_7d, dtype: int, description: d}
""")
    with pytest.raises(RegistryError, match="duplicate feature x_7d"):
        load_registry(p)
