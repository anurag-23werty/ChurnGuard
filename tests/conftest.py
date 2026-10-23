from __future__ import annotations

from pathlib import Path

import pytest

from churnguard.config import DuckDBConfig, IngestConfig, Paths, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    paths = Paths(
        raw=tmp_path / "raw",
        bronze=tmp_path / "bronze",
        silver=tmp_path / "silver",
        gold=tmp_path / "gold",
        reports=tmp_path / "reports",
        models=tmp_path / "models",
    )
    paths.raw.mkdir(parents=True)
    # Tiny blocks / flushes so the tests exercise multi-batch and multi-row-group paths.
    ingest = IngestConfig(csv_block_size=4096, flush_rows=50, validate_sample_rows=100,
                          min_free_disk_gb=0)
    return Settings(paths=paths, kaggle_competition="test", ingest=ingest,
                    duckdb=DuckDBConfig(memory_limit="1GB", threads=2))


def write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


@pytest.fixture
def raw_user_logs(settings: Settings) -> dict[str, int]:
    """v1 spans 2016-11..2017-02 plus one invalid date; v2 is March 2017."""
    header = "msno,date,num_25,num_50,num_75,num_985,num_100,num_unq,total_secs"
    v1 = []
    for month, n in ((201611, 120), (201612, 90), (201701, 60), (201702, 30)):
        v1 += [f"user{i % 7}=,{month}{(i % 28) + 1:02d},1,0,0,0,5,4,1234.5" for i in range(n)]
    v1.append("user1=,20161399,1,0,0,0,5,4,-9.2e15")  # unparseable date, dirty secs
    v2 = [f"user{i % 7}=,201703{(i % 28) + 1:02d},0,1,0,0,3,2,600.0" for i in range(40)]
    write_csv(settings.paths.raw / "user_logs.csv", header, v1)
    write_csv(settings.paths.raw / "user_logs_v2.csv", header, v2)
    return {"v1": len(v1), "v2": len(v2)}
