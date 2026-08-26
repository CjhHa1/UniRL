"""vllm-omni general plugin: teach the model postprocessors about the capture envelope.

Since vllm-omni 0.26 deleted ``DiffusionOutput.custom_output``, the RL
pipelines export their captures on the output envelope
``{"payload": ..., "metadata": ...}`` (see
``pipelines._shared.interception.stamp_capture``). Only Qwen-Image's
postprocess understands that shape; SD3, HunyuanVideo 1.5, HunyuanImage3 and
BAGEL all call ``image_processor.postprocess(images)`` on whatever they are
handed and would choke on a dict.

This plugin wraps whichever postprocess the registry resolves so it unwraps
the envelope, hands the model its bare tensor, and re-attaches the captures to
the result. It runs in the stage process, which is where the postprocess is
resolved — ``prepare_engine_environment`` loads this group before the
``DiffusionEngine`` is built, and neither of unirl's other hooks
(``custom_pipeline_args.pipeline_class``, ``worker_extension_cls``) lands
there.

Registered as a ``vllm_omni.general_plugins`` entry point, so it applies to
every stage process without the stage YAMLs opting in. Loading is idempotent
because plugins may be loaded more than once per process.
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
        # stamp_capture writes exactly one payload key; anything else is a
        # shape this wrapper did not build and is passed through whole.
        raw = next(iter(payload.values())) if len(payload) == 1 else payload

        result = func(raw, **kwargs)
        if not captures:
            return result

        # Reuse upstream's own normalization so legacy payload keys (fps,
        # audio_sample_rate) still become proper metadata groups.
        normalized = normalize_diffusion_postprocess_output(result)
        return {
            "payload": dict(normalized.outputs),
            "metadata": _merge_metadata(dict(normalized.metadata), captures),
        }

    # The engine decides whether to pass ``sampling_params`` with
    # ``_func_accepts_parameter``, which reads ``inspect.signature`` and treats a
    # ``**kwargs`` as "accepts anything". Without the ``__wrapped__`` that
    # ``functools.wraps`` sets, every model's postprocess would look like it
    # takes ``sampling_params`` and the ones that do not would raise on it.
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
