"""vllm-omni general plugin: teach the model postprocessors about the capture envelope."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_unirl_envelope_aware"


def _merge_metadata(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for group, values in extra.items():
        current = merged.get(group)
        if isinstance(current, dict) and isinstance(values, dict):
            merged[group] = {**current, **values}
        else:
            merged[group] = values
    return merged


def _envelope_aware(func: Callable[..., Any]) -> Callable[..., Any]:
    from vllm_omni.diffusion.output_formatter import normalize_diffusion_postprocess_output

    def wrapped(outputs: Any, **kwargs: Any) -> Any:
        if not (isinstance(outputs, dict) and isinstance(outputs.get("payload"), dict)):
            return func(outputs, **kwargs)

        payload = outputs["payload"]
        captures = outputs.get("metadata") or {}
        # BAGEL's upstream payload also carries ``trajectory``; postprocess the
        # generated tensor, not the whole payload dict.
        if "image" in payload:
            raw = payload["image"]
        elif "video" in payload:
            raw = payload["video"]
        elif len(payload) == 1:
            raw = next(iter(payload.values()))
        else:
            raw = payload

        result = func(raw, **kwargs)
        if not captures:
            return result

        # Upstream's normalization turns legacy payload keys into metadata groups.
        normalized = normalize_diffusion_postprocess_output(result)
        return {
            "payload": dict(normalized.outputs),
            "metadata": _merge_metadata(dict(normalized.metadata), captures),
        }

    # The engine picks who gets ``sampling_params`` from ``inspect.signature``, and
    # reads a bare ``**kwargs`` as accepting it; ``__wrapped__`` keeps that honest.
    functools.update_wrapper(wrapped, func)
    setattr(wrapped, _PATCH_FLAG, True)
    return wrapped


def register_envelope_postprocess() -> None:
    """Wrap the resolved diffusion postprocess in every module that uses it."""
    from vllm_omni.diffusion import diffusion_engine, registry

    original = registry.get_diffusion_post_process_func
    if getattr(original, _PATCH_FLAG, False):
        return

    def patched(od_config: Any) -> Any:
        func = original(od_config)
        return None if func is None else _envelope_aware(func)

    setattr(patched, _PATCH_FLAG, True)
    registry.get_diffusion_post_process_func = patched
    # diffusion_engine took a from-import, so it holds its own binding.
    diffusion_engine.get_diffusion_post_process_func = patched
    logger.info("unirl: diffusion postprocess is now capture-envelope aware")
