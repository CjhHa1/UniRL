"""vllm-omni general plugin: teach the model postprocessors about the capture envelope.

vllm-omni 0.26 deleted ``DiffusionOutput.custom_output``, so RL captures now ride
the output envelope (see ``pipelines._shared.interception.stamp_capture``). Only
Qwen-Image's postprocess understands that shape; the other four hand whatever they
get to ``image_processor.postprocess`` and choke on a dict. Registered as a
``vllm_omni.general_plugins`` entry point because that loads in the stage process,
the only place the postprocess can be wrapped before the engine resolves it.
"""

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
        # stamp_capture writes exactly one payload key; any other shape is not ours.
        raw = next(iter(payload.values())) if len(payload) == 1 else payload

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
