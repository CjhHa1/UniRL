"""Resolve stable, filesystem-safe identifiers shared by a Ray run."""

from __future__ import annotations

import os
import re

import ray


def resolve_run_id(explicit: str | None = None) -> str:
    """Return a filesystem-safe id shared by every rank in one run."""
    raw = (explicit or os.environ.get("UNIRL_RUN_ID") or ray.get_runtime_context().get_job_id().hex()).strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not safe:
        raise ValueError(f"run id {raw!r} has no filesystem-safe characters.")
    return safe


__all__ = ["resolve_run_id"]
