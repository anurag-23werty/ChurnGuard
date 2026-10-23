"""DuckDB connection factory with memory/thread limits and an on-disk spill directory."""

from __future__ import annotations

import duckdb

from churnguard.config import Settings, get_settings


def connect(settings: Settings | None = None) -> duckdb.DuckDBPyConnection:
    settings = settings or get_settings()
    spill = settings.paths.gold.parent / ".duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{settings.duckdb.memory_limit}'")
    con.execute(f"SET threads = {settings.duckdb.threads}")
    con.execute(f"SET temp_directory = '{spill}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def bronze_glob(settings: Settings, table: str) -> str:
    return str(settings.paths.bronze / table / "**" / "*.parquet")
