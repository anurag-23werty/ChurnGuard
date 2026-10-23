"""Open raw files as byte streams without extracting them to disk.

user_logs.csv is ~30 GB uncompressed, which does not fit next to everything else on a
laptop. `7zz x -so` decompresses to stdout, and pyarrow reads that pipe incrementally.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

_SEVEN_ZIP_BINARIES = ("7zz", "7z", "7za")


def seven_zip_binary() -> str:
    for name in _SEVEN_ZIP_BINARIES:
        if path := shutil.which(name):
            return path
    raise RuntimeError("7-Zip not found. Install it with: brew install sevenzip")


def resolve_raw_file(raw_dir: Path, archive: str) -> Path:
    """Prefer the .7z archive; fall back to an already-extracted .csv of the same name."""
    archive_path = raw_dir / archive
    if archive_path.exists():
        return archive_path
    csv_path = raw_dir / archive.removesuffix(".7z")
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"neither {archive_path} nor {csv_path} exists (run `make download`)")


@contextmanager
def open_stream(path: Path) -> Iterator[BinaryIO]:
    if path.suffix != ".7z":
        with open(path, "rb") as f:
            yield f
        return

    proc = subprocess.Popen(
        [seven_zip_binary(), "x", "-so", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=16 << 20,
    )
    assert proc.stdout is not None
    try:
        yield proc.stdout
    except BaseException:
        proc.kill()
        raise
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        returncode = proc.wait()
    # Only reached on normal exit: a truncated or corrupt archive must fail the ingest.
    if returncode != 0:
        raise RuntimeError(f"7-Zip failed on {path.name} (exit {returncode}): {stderr.strip()}")
