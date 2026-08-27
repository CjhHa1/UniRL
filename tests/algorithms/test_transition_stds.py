"""Transition-standard-deviation contracts for diffusion replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.algorithms.base import _gaussian_kl_div, _transition_sigma
from unirl.models.sensenova_u1.conditions import SenseNovaU1Conditions
from unirl.models.sensenova_u1.diffusion import SenseNovaU1DiffusionParams, SenseNovaU1DiffusionStage
from unirl.models.types.replay_result import ReplayResult, compute_transition_stds
from unirl.types.segments.latent import LatentSegment


class _FixedStrategy:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def transition_std(self, *, sigma, **kwargs) -> torch.Tensor:
        return torch.full_like(sigma, self.value)


def test_transition_sigma_uses_replay_coordinates_and_packed_rank() -> None:
    means = torch.zeros(3, 2, 7, 11)
    provided = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    replay = ReplayResult(
        log_probs=torch.zeros(3, 2),
        prev_sample_means=means,
        transition_stds=provided,
    )

    resolved = _transition_sigma(
        replay,
        target_steps=[0, 1],
        device=torch.device("cpu"),
        like=means,
    )

    assert resolved.shape == (3, 2, 1, 1)
    torch.testing.assert_close(resolved[..., 0, 0], provided)
    assert _gaussian_kl_div(torch.ones_like(means), means, resolved).shape == means.shape


def test_replay_result_requires_stage_provided_stds() -> None:
    means = torch.zeros(3, 2, 7, 11)

    with pytest.raises(ValueError, match="same-coordinate transition_stds"):
        ReplayResult(
            log_probs=torch.zeros(3, 2),
            prev_sample_means=means,
        )


def test_replay_result_rejects_incompatible_std_tail_shape() -> None:
    with pytest.raises(ValueError, match="not broadcastable"):
        ReplayResult(
            log_probs=torch.zeros(3, 2),
            prev_sample_means=torch.zeros(3, 2, 7, 11),
            transition_stds=torch.zeros(1, 2, 5),
        )


def test_compute_transition_stds_returns_step_aligned_contract() -> None:
    resolved = compute_transition_stds(
        _FixedStrategy(0.5),
        sigmas=torch.tensor([1.0, 0.7, 0.3]),
        step_indices=[0, 1],
        eta=0.7,
        sigma_max=0.7,
    )

    assert resolved.shape == (1, 2)
    torch.testing.assert_close(resolved, torch.full((1, 2), 0.5))


class _ReplayStep:
    def step_with_logp(self, *args, sample, prev_sample, **kwargs):
        return prev_sample, torch.zeros(sample.shape[0]), torch.zeros_like(sample)


def test_sensenova_replay_returns_pixel_space_transition_std() -> None:
    model = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(
            config=SimpleNamespace(t_eps=0.0),
            patch_size=16,
            downsample_ratio=0.5,
            noise_scale=1.0,
            noise_scale_mode="resolution",
            noise_scale_base_image_seq_len=64,
            noise_scale_max_value=16.0,
        ),
    )
    stage = SenseNovaU1DiffusionStage(
        model=model,
        step=_ReplayStep(),
        strategy=_FixedStrategy(0.25),
        autocast_precision="fp32",
    )
    conditions = SenseNovaU1Conditions(
        prompts=["prompt"],
        condition_caches=[None],
        uncondition_caches=[None],
        condition_image_indexes=[None],
        uncondition_image_indexes=[None],
        image_shapes=[(768, 768)],
    )
    segment = LatentSegment(
        latents=torch.zeros(1, 2, 2, 3),
        sigmas=torch.tensor([0.8, 0.6]),
        indices=torch.tensor([0, 1]),
        sde_indices=torch.tensor([0]),
    )

    result = stage.replay(
        conditions,
        segment=segment,
        params=SenseNovaU1DiffusionParams(num_inference_steps=1, eta=0.7),
    )

    assert result.transition_stds is not None
    assert result.transition_stds.shape == (1, 1)
    # At 768², resolution noise_scale is 3, so unit std 0.25 becomes 0.75.
    torch.testing.assert_close(result.transition_stds, torch.tensor([[0.75]]))
