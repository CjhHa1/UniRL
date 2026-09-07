"""Resolve stable, filesystem-safe identifiers shared by a Ray run."""

from __future__ import annotations

import os
import re
from typing import Optional


def resolve_run_id(explicit: Optional[str] = None, *, fallback: Optional[str] = None) -> str:
    """Return a filesystem-safe id shared by every rank in one run."""
    raw = str(explicit or os.environ.get("UNIRL_RUN_ID", "")).strip()
    if not raw:
        try:
            import ray

            if ray.is_initialized():
                job_id = ray.get_runtime_context().get_job_id()
                as_hex = getattr(job_id, "hex", None)
                raw = str(as_hex() if callable(as_hex) else job_id)
        except ImportError:
            pass
    raw = raw or str(fallback or "").strip()
    if not raw:
        raise RuntimeError(
            "A shared run id is required for run-scoped files. Pass run_id or set "
            "UNIRL_RUN_ID when no Ray job context is available."
        )
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not safe:
        raise ValueError(f"run id {raw!r} has no filesystem-safe characters.")
    return safe


__all__ = ["resolve_run_id"]
