from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest

from churnguard.data.sources import resolve_raw_file
from churnguard.storage import MARKER, archive, resolve_layer, restore

needs_rclone = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone not installed")


@pytest.fixture
def local_remote(settings, tmp_path):
    # rclone treats a plain path as a local remote, so tests need no cloud account.
    remote = tmp_path / "remote"
    return replace(settings, storage=replace(settings.storage, remote=str(remote))), remote


def test_resolve_layer_rejects_paths_outside_data(settings):
    for bad in ("", ".", "..", "../etc", "/etc"):
        with pytest.raises(ValueError):
            resolve_layer(settings, bad)
    local, remote = resolve_layer(settings, "bronze/user_logs")
    assert local == (settings.paths.bronze / "user_logs").resolve()
    assert remote.endswith("/data/bronze/user_logs")


@needs_rclone
def test_archive_then_restore_roundtrip(local_remote):
    settings, remote = local_remote
    raw = settings.paths.raw
    (raw / "a.csv.7z").write_bytes(b"x" * 1000)
    (raw / "sub").mkdir()
    (raw / "sub" / "b.bin").write_bytes(b"y" * 10)
    (raw / ".partial").mkdir()
    (raw / ".partial" / "junk").write_bytes(b"z")

    archive(settings, "raw")
    assert sorted(p.name for p in raw.iterdir()) == [MARKER]
    assert (remote / "data" / "raw" / "a.csv.7z").stat().st_size == 1000
    assert not (remote / "data" / "raw" / ".partial").exists()
    assert json.loads((raw / MARKER).read_text())["restore"] == "make restore LAYER=raw"

    with pytest.raises(FileNotFoundError, match="make restore LAYER=raw"):
        resolve_raw_file(raw, "a.csv.7z")

    restore(settings, "raw")
    assert (raw / "a.csv.7z").read_bytes() == b"x" * 1000
    assert (raw / "sub" / "b.bin").exists()
    assert not (raw / MARKER).exists()


@needs_rclone
def test_keep_local_does_not_delete(local_remote):
    settings, remote = local_remote
    (settings.paths.raw / "a.csv.7z").write_bytes(b"x")
    archive(settings, "raw", keep_local=True)
    assert (settings.paths.raw / "a.csv.7z").exists()
    assert (remote / "data" / "raw" / "a.csv.7z").exists()
