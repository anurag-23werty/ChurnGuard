"""Raw archives -> bronze Parquet.

Streams each CSV in blocks (never fully in memory, never extracted to disk), types it,
converts YYYYMMDD ints to dates, validates against the Pandera contract, and writes zstd
Parquet partitioned by month. Every run writes to a staging directory and swaps it in
atomically, with a _manifest.json that reconciles rows read vs. written.

Usage:
  python -m churnguard.data.bronze                                  # all tables
  python -m churnguard.data.bronze --tables user_logs --months 2016-10:2017-03
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from churnguard.config import IngestConfig, Settings, get_settings
from churnguard.data.catalog import TABLES, TableSpec
from churnguard.data.schemas import BY_TABLE
from churnguard.data.sources import open_stream, resolve_raw_file
from churnguard.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

INVALID_PARTITION = "invalid"


@dataclass
class SourceStats:
    rows_read: int = 0
    rows_written: int = 0
    rows_out_of_window: int = 0
    invalid_dates: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    null_required: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def month_key(yyyymm: int) -> str:
    return f"{yyyymm // 100:04d}-{yyyymm % 100:02d}"


def parse_month_window(spec: str | None) -> tuple[int | None, int | None]:
    """'2016-10:2017-03' -> (201610, 201703). Either side may be empty."""
    if not spec:
        return None, None
    start, _, end = spec.partition(":")

    def to_int(s: str) -> int | None:
        return int(s.replace("-", "")) if s else None

    return to_int(start), to_int(end or start)


def yyyymmdd_to_date(col: pa.Array) -> pa.Array:
    ts = pc.strptime(pc.cast(col, pa.string()), format="%Y%m%d", unit="s", error_is_null=True)
    return pc.cast(ts, pa.date32())


class PartitionedWriter:
    """Buffers rows per month partition and flushes them as Parquet row groups.

    One open ParquetWriter per partition (≈30 months for KKBox), so each month ends up
    as a single file per source with large row groups.
    """

    def __init__(self, out_dir: Path, file_stem: str, cfg: IngestConfig, schema: pa.Schema):
        self.out_dir = out_dir
        self.file_stem = file_stem
        self.cfg = cfg
        self.schema = schema
        self._buffers: dict[str, list[pa.Table]] = defaultdict(list)
        self._buffered_rows: dict[str, int] = defaultdict(int)
        self._writers: dict[str, pq.ParquetWriter] = {}
        self.rows_per_partition: dict[str, int] = defaultdict(int)

    def _path(self, partition: str | None) -> Path:
        base = self.out_dir if partition is None else self.out_dir / f"month={partition}"
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{self.file_stem}.parquet"

    def write(self, table: pa.Table, partition: str | None) -> None:
        key = partition or ""
        self._buffers[key].append(table)
        self._buffered_rows[key] += table.num_rows
        self.rows_per_partition[key] += table.num_rows
        if self._buffered_rows[key] >= self.cfg.flush_rows:
            self._flush(key)

    def _flush(self, key: str) -> None:
        if not self._buffers[key]:
            return
        if key not in self._writers:
            self._writers[key] = pq.ParquetWriter(
                self._path(key or None), self.schema, compression=self.cfg.compression
            )
        self._writers[key].write_table(pa.concat_tables(self._buffers[key]))
        self._buffers[key].clear()
        self._buffered_rows[key] = 0

    def close(self) -> None:
        for key in list(self._buffers):
            self._flush(key)
        for w in self._writers.values():
            w.close()


def _output_schema(spec: TableSpec) -> pa.Schema:
    fields = [
        pa.field(f.name, pa.date32()) if f.name in spec.date_columns else f
        for f in spec.read_schema
    ]
    fields.append(pa.field("source", pa.string()))
    extra_cols = {c for extras in spec.extra_by_source.values() for c in extras}
    fields.extend(pa.field(c, pa.string()) for c in sorted(extra_cols))
    return pa.schema(fields)


def _transform(batch: pa.RecordBatch, spec: TableSpec, source: str, stats: SourceStats):
    """Returns (typed table, partition keys as YYYYMM int array or None)."""
    table = pa.Table.from_batches([batch])
    partition_keys = None

    # Checked on raw values: a missing key is a contract breach, whereas a present but
    # unparseable date is routed to the "invalid" partition and counted separately.
    for col in spec.required:
        stats.null_required[col] += table[col].null_count

    for col in spec.date_columns:
        raw = table[col]
        parsed = yyyymmdd_to_date(raw)
        stats.invalid_dates[col] += parsed.null_count - raw.null_count
        if col == spec.partition_by:
            # Rows whose partition date is unparseable get key 0 -> "invalid" partition.
            partition_keys = pc.if_else(pc.is_null(parsed), 0, pc.divide(raw, 100))
        table = table.set_column(table.schema.get_field_index(col), col, parsed)

    n = table.num_rows
    table = table.append_column("source", pa.array([source] * n, pa.string()))
    for c, v in sorted(spec.extra_by_source.get(source, {}).items()):
        table = table.append_column(c, pa.array([v] * n, pa.string()))
    return table, partition_keys


_NULLABLE_INTS = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
}


def _validate_sample(table: pa.Table, spec: TableSpec, sample_rows: int) -> None:
    sample = table.slice(0, sample_rows).to_pandas(types_mapper=_NULLABLE_INTS.get)
    BY_TABLE[spec.name].validate(sample, lazy=True)


def _write_partitioned(
    table: pa.Table,
    keys: pa.ChunkedArray,
    writer: PartitionedWriter,
    window: tuple[int | None, int | None],
    stats: SourceStats,
) -> None:
    start, end = window
    keys = pc.fill_null(keys, 0)
    in_window = pc.and_(
        pc.greater_equal(keys, start) if start else pa.scalar(True),
        pc.less_equal(keys, end) if end else pa.scalar(True),
    )
    # Invalid-date rows are always kept so they show up in the DQ report.
    keep = pc.or_(in_window, pc.equal(keys, 0))
    stats.rows_out_of_window += table.num_rows - pc.sum(pc.cast(keep, pa.int64())).as_py()
    table, keys = table.filter(keep), keys.filter(keep)
    if table.num_rows == 0:
        return

    # Sort once by partition key, then slice contiguous runs.
    order = pc.sort_indices(keys)
    table, keys = table.take(order), keys.take(order).to_numpy()
    boundaries = [0, *((keys[1:] != keys[:-1]).nonzero()[0] + 1).tolist(), len(keys)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
        k = int(keys[lo])
        writer.write(table.slice(lo, hi - lo), INVALID_PARTITION if k == 0 else month_key(k))
        stats.rows_written += hi - lo


def ingest_table(
    spec: TableSpec, settings: Settings, months: str | None = None
) -> dict:
    cfg = settings.ingest
    window = parse_month_window(months) if months else parse_month_window(
        f"{cfg.month_start or ''}:{cfg.month_end or ''}" if cfg.month_start or cfg.month_end
        else None
    )
    final_dir = settings.paths.bronze / spec.name
    staging_dir = settings.paths.bronze / f".staging_{spec.name}"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)

    schema = _output_schema(spec)
    manifest: dict = {
        "table": spec.name,
        "started_at": datetime.now(UTC).isoformat(),
        "month_window": [w and month_key(w) for w in window] if spec.partition_by else None,
        "sources": {},
    }
    t0 = time.monotonic()

    for f in spec.files:
        path = resolve_raw_file(settings.paths.raw, f.archive)
        stats = SourceStats()
        writer = PartitionedWriter(staging_dir, f.source, cfg, schema)
        log.info("[%s] reading %s", spec.name, path.name)

        with open_stream(path) as stream:
            reader = pacsv.open_csv(
                stream,
                read_options=pacsv.ReadOptions(block_size=cfg.csv_block_size),
                convert_options=pacsv.ConvertOptions(
                    column_types=spec.read_schema,
                    include_columns=spec.read_schema.names,
                    strings_can_be_null=True,
                ),
            )
            for i, batch in enumerate(reader):
                stats.rows_read += batch.num_rows
                table, keys = _transform(batch, spec, f.source, stats)
                table = table.cast(schema)
                if i == 0 or i % 25 == 0:
                    _validate_sample(table, spec, cfg.validate_sample_rows)

                if spec.partition_by:
                    _write_partitioned(table, keys, writer, window, stats)
                else:
                    writer.write(table, None)
                    stats.rows_written += table.num_rows

                if i % 50 == 0:
                    rate = stats.rows_read / max(time.monotonic() - t0, 1e-9)
                    log.info("[%s/%s] %s rows read (%.0fk rows/s)",
                             spec.name, f.source, f"{stats.rows_read:,}", rate / 1e3)
        writer.close()

        if stats.rows_read != stats.rows_written + stats.rows_out_of_window:
            raise RuntimeError(f"row reconciliation failed for {path.name}: {stats}")
        bad_required = {k: v for k, v in stats.null_required.items() if v}
        if bad_required:
            raise ValueError(f"{path.name}: nulls in required columns {bad_required}")

        manifest["sources"][f.source] = {
            "file": path.name,
            "rows_read": stats.rows_read,
            "rows_written": stats.rows_written,
            "rows_out_of_window": stats.rows_out_of_window,
            "invalid_dates": dict(stats.invalid_dates),
            "partitions": {k or "_all": v for k, v in sorted(writer.rows_per_partition.items())},
        }
        log.info("[%s/%s] done: %s rows written", spec.name, f.source, f"{stats.rows_written:,}")

    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["duration_s"] = round(time.monotonic() - t0, 1)
    (staging_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Atomic-ish swap: readers never see a half-written table.
    if final_dir.exists():
        backup = final_dir.with_name(f".old_{spec.name}")
        final_dir.replace(backup)
        staging_dir.replace(final_dir)
        shutil.rmtree(backup)
    else:
        staging_dir.replace(final_dir)
    return manifest


def check_free_disk(path: Path, min_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_gb:
        raise RuntimeError(f"only {free_gb:.1f} GB free at {path}; need >= {min_gb} GB")
    log.info("free disk: %.1f GB", free_gb)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Ingest KKBox raw files into bronze Parquet")
    parser.add_argument("--tables", nargs="+", choices=sorted(TABLES), default=list(TABLES))
    parser.add_argument("--months", help="inclusive window YYYY-MM:YYYY-MM (overrides config)")
    args = parser.parse_args(argv)

    settings = get_settings()
    check_free_disk(settings.paths.bronze, settings.ingest.min_free_disk_gb)
    for name in args.tables:
        m = ingest_table(TABLES[name], settings, args.months)
        total = sum(s["rows_written"] for s in m["sources"].values())
        log.info("[%s] committed %s rows in %.1fs", name, f"{total:,}", m["duration_s"])


if __name__ == "__main__":
    main()
