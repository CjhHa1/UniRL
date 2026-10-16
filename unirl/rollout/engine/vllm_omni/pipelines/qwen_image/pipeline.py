"""RL-aware Qwen-Image pipeline subclass."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    drain_trajectory_into,
    finalize_output,
    inject_latents,
    make_sde_scheduler,
    resolve_request_noise,
    single_request,
    stamp_capture,
)


class RLQwenImagePipeline(QwenImagePipeline):
    """Qwen-Image pipeline with the RL interception protocol installed."""

    # Upstream opts into request batching; RL rollout does not — see ``single_request``.
    supports_request_batch = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._upstream_scheduler: FlowMatchEulerDiscreteScheduler = self.scheduler
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        self._pending_initial_noise: Optional[torch.Tensor] = None
        self._harvest_hw: Optional[Tuple[int, int]] = None

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler via ``from_config``; always installed, even at eta=0."""
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        self.scheduler = make_sde_scheduler(self._upstream_scheduler.config)

    def _install_conditioning_tap(self) -> None:
        """Wrap ``encode_prompt`` to capture the text conditioning."""
        if self._conditioning_tap_installed:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            cap = pipeline_self._captured_conditioning
            if cap is not None:
                prompt_embeds, prompt_embeds_mask = result
                if "prompt_embeds" not in cap:
                    cap["prompt_embeds"] = detach_cpu(prompt_embeds)
                    cap["prompt_embeds_mask"] = detach_cpu(prompt_embeds_mask)
                elif "negative_prompt_embeds" not in cap:
                    cap["negative_prompt_embeds"] = detach_cpu(prompt_embeds)
                    cap["negative_prompt_embeds_mask"] = detach_cpu(prompt_embeds_mask)
            return result

        self.encode_prompt = tapped  # type: ignore[assignment]
        self._conditioning_tap_installed = True

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection: the driver's ``[B, C, H, W]`` x_T is packed to ``[B, S, C*4]``. Consume-once."""
        upstream = super().prepare_latents
        noise = self._pending_initial_noise
        if noise is not None:
            self._pending_initial_noise = None
            if len(args) < 4:
                raise RuntimeError(
                    "RLQwenImagePipeline.prepare_latents: expected upstream's "
                    f"fully positional prepare_latents call; got {len(args)} positional args."
                )
            batch, channels = args[0], args[1]
            grid_h = 2 * (args[2] // (self.vae_scale_factor * 2))
            grid_w = 2 * (args[3] // (self.vae_scale_factor * 2))
            if tuple(noise.shape) != (batch, channels, grid_h, grid_w):
                raise RuntimeError(
                    "RLQwenImagePipeline: driver x_T shape "
                    f"{tuple(noise.shape)} does not match the worker latent grid "
                    f"[{batch}, {channels}, {grid_h}, {grid_w}] for "
                    f"{args[2]}x{args[3]} px — check the recipe's "
                    "init_noise_latent_shape / initial_noise_batch."
                )
            packed = self._pack_latents(noise, batch, channels, grid_h, grid_w)
            args, kwargs = inject_latents(upstream, args, kwargs, packed)
        return upstream(*args, **kwargs)

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> list[DiffusionOutput]:
        """Single-request batch in, one-element list out — see ``single_request``."""
        one = single_request(req, caller="RLQwenImagePipeline.forward")
        self._install_sde_scheduler()
        self._install_conditioning_tap()

        extra = one.sampling_params.extra_args
        self.scheduler.arm(eta=one.sampling_params.eta, sde_indices=extra.get("sde_indices"))
        self._pending_initial_noise = resolve_request_noise(one, caller="RLQwenImagePipeline.forward")
        self._captured_conditioning = {}
        height = one.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = one.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        self._harvest_hw = (height, width)

        outs = super().forward(req, **kwargs)

        out = outs[0]
        drain_trajectory_into(out, self.scheduler)
        if out.trajectory_latents is not None:
            packed = out.trajectory_latents
            b, t1 = packed.shape[0], packed.shape[1]
            flat = self._unpack_latents(
                packed.reshape(b * t1, *packed.shape[2:]),
                self._harvest_hw[0],
                self._harvest_hw[1],
                self.vae_scale_factor,
            )
            flat = flat.squeeze(2)
            out.trajectory_latents = flat.reshape(b, t1, *flat.shape[1:])
        if self._captured_conditioning:
            stamp_capture(out, "text_capture", self._captured_conditioning)
        finalize_output(out)
        return outs


__all__ = ["RLQwenImagePipeline"]
