"""Download KKBox raw archives from Kaggle into data/raw.

Files are fetched one at a time (not the whole competition zip) so we never hold the
bundle and its extracted copy on disk at once. Requires:
  1. an API token in ~/.kaggle/access_token or KAGGLE_API_TOKEN
     (legacy ~/.kaggle/kaggle.json also works), and
  2. the competition rules accepted on kaggle.com.

Usage:
  python -m churnguard.data.download                    # all tables
  python -m churnguard.data.download --tables members labels
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from churnguard.config import get_settings
from churnguard.data.catalog import TABLES
from churnguard.logging_utils import get_logger, setup_logging

log = get_logger(__name__)


def _kaggle_bin() -> str:
    exe = Path(sys.executable).with_name("kaggle")
    return str(exe) if exe.exists() else "kaggle"


def list_remote_files(competition: str) -> list[str]:
    out = subprocess.run(
        [_kaggle_bin(), "competitions", "files", competition, "--csv", "--page-size", "200"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # First column is the file name; skip the CSV header row.
    rows = list(csv.reader(io.StringIO(out.strip())))
    return [row[0] for row in rows[1:] if row]


def _resolve_remote_name(archive: str, remote: list[str]) -> str:
    """Kaggle may nest files under folders; match on basename."""
    matches = [r for r in remote if Path(r).name == archive]
    if not matches:
        raise FileNotFoundError(f"{archive} not found in competition files")
    return matches[0]


def download_file(competition: str, remote_name: str, dest: Path) -> Path:
    target = dest / Path(remote_name).name
    if target.exists():
        log.info("skip %s (already present, %.1f MB)", target.name, target.stat().st_size / 1e6)
        return target

    # Download into a staging dir and move on success, so an interrupted run never leaves
    # a truncated archive that the "already present" check would then trust.
    staging = dest / ".partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    log.info("downloading %s", remote_name)
    subprocess.run(
        [_kaggle_bin(), "competitions", "download", competition, "-f", remote_name,
         "-p", str(staging), "-q"],
        check=True,
    )

    staged = staging / target.name
    # Kaggle sometimes wraps single files in a zip; unwrap and drop the zip to save disk.
    wrapped = staging / f"{target.name}.zip"
    if not staged.exists() and wrapped.exists():
        with zipfile.ZipFile(wrapped) as zf:
            member = next(m for m in zf.namelist() if Path(m).name == target.name)
            with zf.open(member) as src, open(staged, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 << 20)
        wrapped.unlink()

    if not staged.exists():
        raise RuntimeError(f"download finished but {staged.name} is missing in {staging}")
    staged.replace(target)
    shutil.rmtree(staging, ignore_errors=True)
    log.info("saved %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
    return target


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", nargs="+", choices=sorted(TABLES), default=sorted(TABLES))
    args = parser.parse_args(argv)

    settings = get_settings()
    raw_dir = settings.paths.raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    remote = list_remote_files(settings.kaggle_competition)
    for table in args.tables:
        for f in TABLES[table].files:
            download_file(
                settings.kaggle_competition, _resolve_remote_name(f.archive, remote), raw_dir
            )


if __name__ == "__main__":
    main()
