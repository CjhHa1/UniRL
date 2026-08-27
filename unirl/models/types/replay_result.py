"""Structured return type for trainable-stage replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


@dataclass
class ReplayResult:
    """Per-stage replay output."""

    log_probs: torch.Tensor
    """Aligned with ``segment.sde_logp`` (or its slice when ``step_indices``
    subsets). Shape ``[B, S']`` for diffusion replay."""

    prev_sample_means: Optional[torch.Tensor] = None
    """The SDE transition's mean μ_θ at each replayed step. Shape
    ``[B, S', *latent_shape]`` for diffusion. Used by GRPO's KL penalty.
    ``None`` when the stage doesn't produce it."""

    transition_stds: Optional[torch.Tensor] = None
    """The SDE transition standard deviation aligned with
    :attr:`prev_sample_means`, in the same coordinate system. Shape ``[B, S']``
    or any shape broadcastable to ``[B, S', *latent_shape]``. Stages that
    return diffusion transition means must populate this field."""

    logits: Optional[torch.Tensor] = None
    """Per-step token logits at each replayed position. Shape
    ``[B, S', V]`` for AR. Reserved for future full-categorical KL
    or entropy penalty support; not needed for Binary KL (which uses
    only per-token log-probs). Currently not populated."""

    values: Optional[torch.Tensor] = None
    """Per-token critic predictions ``V_t``. Packed ``[total_tokens]`` for AR.
    ``None`` when replay did not request a value head."""

    def __post_init__(self) -> None:
        """Enforce the complete stochastic-diffusion replay contract."""
        if self.prev_sample_means is None:
            return
        if self.transition_stds is None:
            raise ValueError("ReplayResult: prev_sample_means requires same-coordinate transition_stds.")
        if self.prev_sample_means.ndim < 2:
            raise ValueError(
                "ReplayResult: prev_sample_means must have shape [B, S, ...]; "
                f"got {tuple(self.prev_sample_means.shape)}."
            )
        batch, steps = self.prev_sample_means.shape[:2]
        if (
            self.transition_stds.ndim < 2
            or int(self.transition_stds.shape[0]) not in {1, int(batch)}
            or int(self.transition_stds.shape[1]) != int(steps)
        ):
            raise ValueError(
                "ReplayResult: transition_stds must align as [B|1, S, ...] with "
                f"prev_sample_means; got stds={tuple(self.transition_stds.shape)}, "
                f"means={tuple(self.prev_sample_means.shape)}."
            )
        if self.transition_stds.ndim > self.prev_sample_means.ndim:
            raise ValueError(
                "ReplayResult: transition_stds rank cannot exceed prev_sample_means "
                f"rank; got stds={tuple(self.transition_stds.shape)}, "
                f"means={tuple(self.prev_sample_means.shape)}."
            )
        padded_std_shape = (
            *self.transition_stds.shape,
            *([1] * (self.prev_sample_means.ndim - self.transition_stds.ndim)),
        )
        try:
            torch.broadcast_shapes(padded_std_shape, tuple(self.prev_sample_means.shape))
        except RuntimeError as exc:
            raise ValueError(
                "ReplayResult: transition_stds is not broadcastable to "
                f"prev_sample_means after adding trailing singleton dimensions; "
                f"got stds={tuple(self.transition_stds.shape)}, "
                f"means={tuple(self.prev_sample_means.shape)}."
            ) from exc


def compute_transition_stds(
    strategy: Any,
    *,
    sigmas: torch.Tensor,
    step_indices: Sequence[int],
    eta: float,
    sigma_max: float | torch.Tensor,
) -> torch.Tensor:
    """Compute schedule-aligned SDE standard deviations as ``[1, S]``."""
    transition_std = getattr(strategy, "transition_std", None)
    if not callable(transition_std):
        raise TypeError(
            f"{type(strategy).__name__} cannot replay stochastic transitions without a transition_std() method."
        )
    indices = torch.as_tensor(step_indices, dtype=torch.long, device=sigmas.device)
    sigma = sigmas.to(dtype=torch.float32).index_select(0, indices)
    sigma_next = sigmas.to(dtype=torch.float32).index_select(0, indices + 1)
    stds = transition_std(
        sigma=sigma,
        sigma_next=sigma_next,
        eta=float(eta),
        sigma_max=sigma_max,
    ).to(device=sigmas.device, dtype=torch.float32)
    if stds.numel() != len(step_indices):
        raise ValueError(
            "compute_transition_stds expected one standard deviation per replayed "
            f"step, got shape {tuple(stds.shape)} for {len(step_indices)} step(s)."
        )
    return stds.reshape(1, len(step_indices))


__all__ = ["ReplayResult", "compute_transition_stds"]
