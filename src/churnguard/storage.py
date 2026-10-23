"""Cold storage for large data layers via rclone (Google Drive, S3, GCS, ... any remote).

Hot data (silver/gold) stays on local disk; cold data (raw archives, bronze) lives on the
remote and is restored on demand. `archive` only deletes local files after `rclone check`
has verified every file on the remote by size and hash.

Usage:
  python -m churnguard.storage status
  python -m churnguard.storage archive raw                 # upload, verify, free local disk
  python -m churnguard.storage archive silver --keep-local # backup only
  python -m churnguard.storage restore bronze/user_logs
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from churnguard.config import Settings, get_settings
from churnguard.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

MARKER = ".ARCHIVED.json"
# Pipeline scratch dirs are never worth archiving.
EXCLUDES = (".partial/**", ".staging_*/**", ".old_*/**", ".duckdb_tmp/**", MARKER)


def _rclone() -> str:
    if exe := shutil.which("rclone"):
        return exe
    raise RuntimeError("rclone not found. Install it with: brew install rclone")


def _run(args: list[str]) -> None:
    log.info("$ rclone %s", " ".join(args))
    try:
        subprocess.run([_rclone(), *args], check=True)
    except subprocess.CalledProcessError as e:
        # The OAuth app stays in Google "Testing" mode, whose refresh tokens expire after
        # 7 days; that is the most common cause of a failure here.
        remote = next((a.split(":")[0] for a in args if ":" in a and not a.startswith("/")), "")
        hint = f" If the token expired, run: rclone config reconnect {remote}:" if remote else ""
        raise RuntimeError(f"rclone {args[0]} failed (exit {e.returncode}).{hint}") from e


def data_root(settings: Settings) -> Path:
    return settings.paths.raw.parent


def resolve_layer(settings: Settings, layer: str) -> tuple[Path, str]:
    """'bronze/user_logs' -> (local path, remote path). Refuses paths outside data/."""
    root = data_root(settings).resolve()
    local = (root / layer).resolve()
    if local == root or root not in local.parents:
        raise ValueError(f"layer must be a path inside {root}, got {layer!r}")
    rel = local.relative_to(root).as_posix()
    return local, f"{settings.storage.remote.rstrip('/')}/data/{rel}"


def _excluded(rel: Path) -> bool:
    return rel.name == MARKER or any(
        part.startswith((".partial", ".staging_", ".old_", ".duckdb_tmp")) for part in rel.parts
    )


def _filters() -> list[str]:
    return [arg for pattern in EXCLUDES for arg in ("--exclude", pattern)]


def archive(settings: Settings, layer: str, keep_local: bool = False) -> None:
    local, remote = resolve_layer(settings, layer)
    files = [p for p in local.rglob("*") if p.is_file() and not _excluded(p.relative_to(local))]
    if not files:
        raise FileNotFoundError(f"nothing to archive at {local}")
    size_gb = sum(p.stat().st_size for p in files) / 1e9
    log.info("archiving %s (%d files, %.2f GB) -> %s", local, len(files), size_gb, remote)

    _run(["copy", str(local), remote, "--transfers", str(settings.storage.transfers),
          "--stats-one-line", "--stats", "30s", *_filters()])
    # --one-way: every local file must exist on the remote with matching size + hash.
    _run(["check", str(local), remote, "--one-way", *_filters()])

    if keep_local:
        log.info("verified; keeping local copy")
        return

    for p in local.iterdir():
        if p.name == MARKER:
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    (local / MARKER).write_text(json.dumps({
        "remote": remote,
        "archived_at": datetime.now(UTC).isoformat(),
        "files": len(files),
        "size_gb": round(size_gb, 3),
        "restore": f"make restore LAYER={layer}",
    }, indent=2))
    log.info("verified and freed %.2f GB locally", size_gb)


def restore(settings: Settings, layer: str) -> None:
    local, remote = resolve_layer(settings, layer)
    local.mkdir(parents=True, exist_ok=True)
    log.info("restoring %s -> %s", remote, local)
    _run(["copy", remote, str(local), "--transfers", str(settings.storage.transfers),
          "--stats-one-line", "--stats", "30s", *_filters()])
    _run(["check", remote, str(local), "--one-way", *_filters()])
    (local / MARKER).unlink(missing_ok=True)
    log.info("restored and verified %s", local)


def archived_hint(path: Path) -> str | None:
    """If `path` sits in an archived layer, return the command that restores it."""
    for parent in (path, *path.parents):
        marker = parent / MARKER
        if marker.exists():
            return json.loads(marker.read_text())["restore"]
    return None


def status(settings: Settings) -> None:
    root = data_root(settings)
    for layer in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for d in (layer, *sorted(p for p in layer.iterdir() if p.is_dir())):
            marker = d / MARKER
            if marker.exists():
                info = json.loads(marker.read_text())
                print(f"{d.relative_to(root)!s:<28} ARCHIVED  {info['size_gb']:>7.2f} GB  "
                      f"-> {info['remote']}")
            elif d == layer:
                size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e9
                print(f"{d.relative_to(root)!s:<28} local     {size:>7.2f} GB")
    print(f"\nfree disk: {shutil.disk_usage(root).free / 1e9:.1f} GB")


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Archive/restore data layers via rclone")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("archive")
    a.add_argument("layer", help="path under data/, e.g. raw, bronze, bronze/user_logs")
    a.add_argument("--keep-local", action="store_true", help="upload + verify, but keep files")
    r = sub.add_parser("restore")
    r.add_argument("layer")
    sub.add_parser("status")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.cmd == "archive":
        archive(settings, args.layer, args.keep_local)
    elif args.cmd == "restore":
        restore(settings, args.layer)
    else:
        status(settings)


if __name__ == "__main__":
    main()
