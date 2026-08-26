"""Shared interception mechanics for the worker-side RL pipelines."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import torch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.types.noise_recipe import NoiseRecipe


def detach_cpu(t: Any) -> Any:
    """Detach + move to CPU for IPC transport. ``None``/non-tensor passthrough."""
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu")
    return t


def detach_cpu_pair(p: Any) -> Any:
    """``(cos, sin)`` rope-cache pair handler. Pass-through otherwise."""
    if isinstance(p, tuple) and len(p) == 2:
        return (detach_cpu(p[0]), detach_cpu(p[1]))
    return p


#: Metadata group namespacing every unirl capture. vllm-omni validates only
#: its own groups (video/audio/actions/transfer/text/common) and tolerates
#: unknown ones, so a private group cannot collide with upstream.
CAPTURE_GROUP = "unirl"


def single_request(req: Any, *, caller: str) -> Any:
    """The one request of a ``DiffusionRequestBatch``.

    vllm-omni 0.26 made ``forward`` take a batch unconditionally, so every RL
    pipeline is handed a ``DiffusionRequestBatch`` even on the single-request
    path. The RL pipelines all declare ``supports_request_batch = False``,
    which makes the engine reject ``max_num_seqs > 1`` at boot and guarantees
    this batch holds exactly one request — the invariant per-request
    trajectory capture depends on. Assert it rather than trust it.
    """
    requests = getattr(req, "requests", None)
    if requests is None:
        return req
    if len(requests) != 1:
        raise RuntimeError(
            f"{caller}: expected a single-request batch (supports_request_batch=False), got {len(requests)}. "
            "Set max_num_seqs=1 on this stage."
        )
    return requests[0]


def stamp_capture(out: Any, key: str, value: Any, *, payload_key: str = "image") -> None:
    """Export a capture on ``DiffusionOutput.output``'s metadata envelope.

    vllm-omni 0.26 deleted ``DiffusionOutput.custom_output`` (#4922) and moved
    pipeline-specific payloads onto the output envelope
    ``{"payload": ..., "metadata": ...}``; the metadata reaches the driver as
    ``OmniRequestOutput.multimodal_output["metadata"]``. Upstream migrated its
    own RL-with-logprob pipeline this exact way in the same commit.

    Upstream pipelines return a bare tensor in ``output``, so the first stamp
    wraps it into the envelope under ``payload_key`` (``"video"`` for hv15).
    That wrap is what ``unirl_postprocess_shim`` unwraps before handing the
    tensor to the model's own postprocess, which is tensor-only for every
    family except Qwen-Image.
    """
    envelope = out.output
    if not (isinstance(envelope, dict) and isinstance(envelope.get("payload"), dict)):
        envelope = {"payload": {payload_key: envelope}}
        out.output = envelope
    metadata = envelope.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError(f"stamp_capture: output['metadata'] must be a dict, got {type(metadata).__name__}")
    metadata.setdefault(CAPTURE_GROUP, {})[key] = value


def set_payload(out: Any, value: Any, *, payload_key: str = "image") -> None:
    """Replace the generated payload without dropping stamped captures.

    Assigning ``out.output`` directly after a :func:`stamp_capture` would
    discard the envelope the captures live on.
    """
    envelope = out.output
    if isinstance(envelope, dict) and isinstance(envelope.get("payload"), dict):
        envelope["payload"][payload_key] = value
    else:
        out.output = value


def read_captures(result: Any) -> Dict[str, Any]:
    """Driver-side inverse of :func:`stamp_capture`.

    Reads the unirl group out of ``OmniRequestOutput.multimodal_output``,
    which is where vllm-omni deposits the envelope's metadata.
    """
    mm = getattr(result, "multimodal_output", None) or {}
    if not isinstance(mm, dict):
        return {}
    metadata = mm.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {}
    captures = metadata.get(CAPTURE_GROUP) or {}
    return captures if isinstance(captures, dict) else {}


def drain_trajectory_into(out: Any, scheduler: Any, *, payload_key: str = "image") -> None:
    """Harvest the SDE scheduler's per-request recordings onto the wire.

    ``trajectory_timesteps`` carries the **true [0, 1] sigma schedule** —
    what the driver reads back as ``LatentSegment.sigmas`` and what replay
    indexes per step. The original 1000-scale per-step timesteps are dropped
    (trivially regenerable from σ). The real sparse SDE step ids ride
    the unirl group's ``sde_step_indices`` so the response layer can echo
    them as the segment's ``sde_indices``.
    """
    traj = scheduler.drain_trajectory()
    if traj is None:
        return
    latents, sigmas, _timesteps, log_probs = traj
    out.trajectory_latents = latents
    out.trajectory_timesteps = sigmas
    out.trajectory_log_probs = log_probs
    stamp_capture(out, "sde_step_indices", scheduler.last_sde_step_indices, payload_key=payload_key)


def _grouped_span(idx: int, spp: int) -> tuple[int, int]:
    spp = int(spp or 1)
    if spp < 1:
        raise ValueError(f"_grouped_span: spp must be >= 1, got {spp}")
    start = int(idx) * spp
    return start, start + spp


def resolve_request_noise(req: Any, *, caller: str) -> Optional[torch.Tensor]:
    """This request's driver x_T, sliced from ``[B, ...]`` keeping its leading ``[num_outputs_per_prompt, ...]``."""
    extra = getattr(req.sampling_params, "extra_args", None) or {}
    noise_batch = extra.get("initial_noise_batch")
    recipe_gids = extra.get("init_noise_group_ids")
    if noise_batch is None and not recipe_gids:
        return None

    rid = str(getattr(req, "request_id", "") or "")
    try:
        idx = int(rid.split("_", 1)[0])
    except ValueError:
        raise RuntimeError(
            f"{caller}: cannot parse batch index from request_id={rid!r}. Expected Omni's ``f'{{i}}_{{uuid}}'`` shape."
        )

    spp = int(getattr(req.sampling_params, "num_outputs_per_prompt", 1) or 1)
    start, end = _grouped_span(idx, spp)

    if noise_batch is not None:
        if start < 0 or end > int(noise_batch.shape[0]):
            raise IndexError(
                f"{caller}: grouped slice [{start}:{end}) out of bounds for "
                f"noise_batch.shape[0]={int(noise_batch.shape[0])}."
            )
        return noise_batch[start:end].clone()

    if start < 0 or end > len(recipe_gids):
        raise IndexError(
            f"{caller}: grouped slice [{start}:{end}) out of bounds for init_noise_group_ids len={len(recipe_gids)}."
        )
    return NoiseRecipe(
        noise_group_ids=[str(g) for g in recipe_gids[start:end]],
        base_seed=int(extra.get("init_noise_seed", 0)),
        latent_shape=tuple(extra["init_noise_latent_shape"]),
    ).resolve()


def inject_latents(
    target: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    noise: torch.Tensor,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Slot a pre-computed x_T into a ``prepare_latents`` call site.

    Upstream calls ``prepare_latents`` with **all args positional**, so the
    slots are read off *target*'s own signature rather than assumed: the three
    families no longer agree on them. vllm-omni 0.27 dropped ``dtype`` and
    ``device`` from sd3's signature, moving ``latents`` from slot 7 to 5, while
    hv15 and qwen-image still carry all three.

    Writing ``kwargs["latents"]`` while ``latents`` is already positional
    raises ``TypeError: got multiple values for argument 'latents'`` — so
    replace the positional slot in place, and fall back to the keyword only for
    partial call shapes. Where the call site still names a dtype/device, the
    noise is moved onto them first; sd3 does that move itself now.
    """
    names = [
        name
        for name, p in inspect.signature(target).parameters.items()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    def _at(name: str) -> Any:
        if name in names:
            idx = names.index(name)
            if len(args) > idx:
                return args[idx]
        return kwargs.get(name)

    dtype, device = _at("dtype"), _at("device")
    if dtype is not None:
        noise = noise.to(dtype=dtype)
    if device is not None:
        noise = noise.to(device=device)

    latents_idx = names.index("latents") if "latents" in names else None
    if latents_idx is not None and len(args) > latents_idx:
        args = (*args[:latents_idx], noise, *args[latents_idx + 1 :])
    else:
        kwargs = {**kwargs, "latents": noise}
    return args, kwargs


def make_sde_scheduler(upstream_config: Any, *, eta: float = 0.0) -> FlowMatchSDEDiscreteScheduler:
    """Build the trajectory-capturing scheduler from the upstream scheduler's config — the sd3/hv15 install path."""
    return FlowMatchSDEDiscreteScheduler.from_config(upstream_config, eta=float(eta))


__all__ = [
    "CAPTURE_GROUP",
    "detach_cpu",
    "detach_cpu_pair",
    "drain_trajectory_into",
    "_grouped_span",
    "inject_latents",
    "make_sde_scheduler",
    "read_captures",
    "resolve_request_noise",
    "set_payload",
    "single_request",
    "stamp_capture",
]
