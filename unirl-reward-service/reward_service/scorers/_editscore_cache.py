"""Atomic, provenance-checked LoRA merge cache for EditScore's vLLM backends."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_CACHE_FORMAT_VERSION = 1
_MANIFEST_NAME = ".unirl-editscore-cache.json"
_STALE_TEMP_AGE_S = 24 * 60 * 60
_MODEL_CLASS_NAMES = {
    "qwen25vl_vllm": "Qwen2_5_VLForConditionalGeneration",
    "qwen3vl_vllm": "Qwen3VLForConditionalGeneration",
}


def _manifest(model_name_or_path: str, lora_path: str, backbone: str) -> dict[str, object]:
    return {
        "format_version": _CACHE_FORMAT_VERSION,
        "backbone": backbone,
        "model_name_or_path": model_name_or_path,
        "lora_path": lora_path,
    }


def _default_cache_dir(manifest: dict[str, object]) -> Path:
    import torch

    identity = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(identity).hexdigest()[:12]
    model_name = Path(str(manifest["model_name_or_path"])).name
    lora_name = Path(str(manifest["lora_path"])).name
    return Path(torch.hub.get_dir()) / "EditScore" / f"{model_name}_{lora_name}_{digest}"


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _validate_cache(cache_dir: Path, expected: dict[str, object]) -> None:
    manifest_path = cache_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            f"EditScore cache {cache_dir} has no {_MANIFEST_NAME}; "
            "use an empty cache path so UniRL can build and publish it atomically"
        )
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"EditScore cache manifest is unreadable: {manifest_path}") from exc
    if actual != expected:
        raise ValueError(
            f"EditScore cache {cache_dir} was built for {actual!r}, not {expected!r}; "
            "choose a cache path matching the configured base model and LoRA"
        )


def _remove_stale_temporaries(target: Path) -> None:
    cutoff = time.time() - _STALE_TEMP_AGE_S
    for temporary in target.parent.glob(f".{target.name}.tmp-*"):
        try:
            if temporary.stat().st_mtime < cutoff:
                shutil.rmtree(temporary, ignore_errors=True)
        except OSError:
            pass


def _merge_checkpoint(
    destination: Path,
    *,
    model_name_or_path: str,
    lora_path: str,
    backbone: str,
) -> None:
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor

    model_cls = getattr(transformers, _MODEL_CLASS_NAMES[backbone])
    model = model_cls.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
    model.save_pretrained(destination)
    AutoProcessor.from_pretrained(model_name_or_path).save_pretrained(destination)


def prepare_merged_checkpoint(
    *,
    model_name_or_path: str,
    lora_path: str,
    backbone: str,
    cache_dir: str | None,
) -> str:
    """Return a complete merged checkpoint, building it once across all ranks."""
    if backbone not in _MODEL_CLASS_NAMES:
        raise ValueError(f"LoRA cache merge requires a vLLM backbone, got {backbone!r}")

    expected = _manifest(model_name_or_path, lora_path, backbone)
    target = Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir(expected)
    lock_path = target.parent / f".{target.name}.lock"

    # A complete cache is immutable, so readers do not need write access to
    # the shared parent merely to validate and use it.
    if target.exists():
        _validate_cache(target, expected)
        return str(target)

    with _exclusive_lock(lock_path):
        if target.exists():
            _validate_cache(target, expected)
            return str(target)

        target.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_temporaries(target)
        temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            temporary.mkdir()
            _merge_checkpoint(
                temporary,
                model_name_or_path=model_name_or_path,
                lora_path=lora_path,
                backbone=backbone,
            )
            manifest_path = temporary / _MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(expected, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                os.replace(temporary, target)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not target.exists():
                    raise
                # A filesystem without cross-host flock semantics may let two
                # builders race. Reuse the winner only if its manifest matches.
                _validate_cache(target, expected)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    return str(target)
