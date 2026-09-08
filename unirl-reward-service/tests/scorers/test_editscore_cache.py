from __future__ import annotations

import errno
import json
import os
import time
from pathlib import Path

import pytest
from reward_service.scorers import _editscore_cache


def test_cache_is_built_once_and_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def merge(destination: Path, **kwargs) -> None:
        calls.append(kwargs)
        (destination / "config.json").write_text("{}")

    monkeypatch.setattr(_editscore_cache, "_merge_checkpoint", merge)
    cache_dir = tmp_path / "merged"
    args = {
        "model_name_or_path": "Qwen/base",
        "lora_path": "EditScore/adapter",
        "backbone": "qwen3vl_vllm",
        "cache_dir": str(cache_dir),
    }

    first = _editscore_cache.prepare_merged_checkpoint(**args)
    second = _editscore_cache.prepare_merged_checkpoint(**args)

    assert first == second == str(cache_dir)
    assert len(calls) == 1
    manifest = json.loads((cache_dir / _editscore_cache._MANIFEST_NAME).read_text())
    assert manifest["model_name_or_path"] == "Qwen/base"
    assert manifest["lora_path"] == "EditScore/adapter"


def test_cache_rejects_wrong_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        _editscore_cache,
        "_merge_checkpoint",
        lambda destination, **_kwargs: (destination / "config.json").write_text("{}"),
    )
    cache_dir = tmp_path / "merged"
    common = {
        "model_name_or_path": "Qwen/base",
        "backbone": "qwen3vl_vllm",
        "cache_dir": str(cache_dir),
    }
    _editscore_cache.prepare_merged_checkpoint(lora_path="EditScore/first", **common)

    with pytest.raises(ValueError, match="not"):
        _editscore_cache.prepare_merged_checkpoint(lora_path="EditScore/second", **common)


def test_valid_cache_does_not_require_lock_file_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        _editscore_cache,
        "_merge_checkpoint",
        lambda destination, **_kwargs: (destination / "config.json").write_text("{}"),
    )
    cache_dir = tmp_path / "merged"
    args = {
        "model_name_or_path": "Qwen/base",
        "lora_path": "EditScore/adapter",
        "backbone": "qwen3vl_vllm",
        "cache_dir": str(cache_dir),
    }
    _editscore_cache.prepare_merged_checkpoint(**args)
    monkeypatch.setattr(
        _editscore_cache,
        "_exclusive_lock",
        lambda _path: pytest.fail("valid cache must not acquire a writer lock"),
    )

    assert _editscore_cache.prepare_merged_checkpoint(**args) == str(cache_dir)


def test_cross_host_publication_race_reuses_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "merged"
    expected = _editscore_cache._manifest("Qwen/base", "EditScore/adapter", "qwen3vl_vllm")

    def merge(destination: Path, **_kwargs) -> None:
        (destination / "config.json").write_text("{}")

    def lose_publication_race(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / _editscore_cache._MANIFEST_NAME).write_text(
            json.dumps(expected),
        )
        raise OSError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination))

    monkeypatch.setattr(_editscore_cache, "_merge_checkpoint", merge)
    monkeypatch.setattr(_editscore_cache.os, "replace", lose_publication_race)

    assert _editscore_cache.prepare_merged_checkpoint(
        model_name_or_path="Qwen/base",
        lora_path="EditScore/adapter",
        backbone="qwen3vl_vllm",
        cache_dir=str(cache_dir),
    ) == str(cache_dir)
    assert not list(tmp_path.glob(".merged.tmp-*"))


def test_partial_legacy_cache_is_not_trusted(tmp_path: Path) -> None:
    cache_dir = tmp_path / "merged"
    cache_dir.mkdir()
    (cache_dir / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="no .unirl-editscore-cache.json"):
        _editscore_cache.prepare_merged_checkpoint(
            model_name_or_path="Qwen/base",
            lora_path="EditScore/adapter",
            backbone="qwen3vl_vllm",
            cache_dir=str(cache_dir),
        )


def test_failed_merge_is_not_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(destination: Path, **_kwargs) -> None:
        (destination / "partial").write_text("incomplete")
        raise RuntimeError("merge failed")

    monkeypatch.setattr(_editscore_cache, "_merge_checkpoint", fail)
    cache_dir = tmp_path / "merged"

    with pytest.raises(RuntimeError, match="merge failed"):
        _editscore_cache.prepare_merged_checkpoint(
            model_name_or_path="Qwen/base",
            lora_path="EditScore/adapter",
            backbone="qwen3vl_vllm",
            cache_dir=str(cache_dir),
        )

    assert not cache_dir.exists()
    assert not list(tmp_path.glob(".merged.tmp-*"))


def test_stale_temp_cleanup_never_blocks_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "merged"
    stale = tmp_path / ".merged.tmp-stale"
    stale.mkdir()
    old = time.time() - _editscore_cache._STALE_TEMP_AGE_S - 1
    os.utime(stale, (old, old))

    def deny_cleanup(*_args, **_kwargs) -> None:
        raise PermissionError("read-only stale directory")

    monkeypatch.setattr(_editscore_cache.shutil, "rmtree", deny_cleanup)

    _editscore_cache._remove_stale_temporaries(target)
