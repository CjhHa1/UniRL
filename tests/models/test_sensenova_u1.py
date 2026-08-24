"""Focused CPU tests for SenseNova-U1.5 geometry and flow conventions."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from unirl.models.sensenova_u1.diffusion import SenseNovaU1DiffusionStep
from unirl.models.sensenova_u1.pipeline import SenseNovaU1Pipeline
from unirl.models.sensenova_u1.pixels import packed_pixel_shape, patchify_pixels, unpatchify_pixels
from unirl.sde.kernels import FlowSDEStrategy


def test_pixel_pack_roundtrip() -> None:
    pixels = torch.arange(3 * 64 * 96, dtype=torch.float32).reshape(1, 3, 64, 96)
    packed = patchify_pixels(pixels, patch_size=32)

    assert packed.shape == (1, *packed_pixel_shape((64, 96), patch_size=32))
    torch.testing.assert_close(
        unpatchify_pixels(packed, image_shape=(64, 96), patch_size=32),
        pixels,
    )


def test_driver_noise_uses_upstream_nchw_layout() -> None:
    sampling = SimpleNamespace(height=512, width=768)

    assert SenseNovaU1Pipeline.latent_shape(model_config=None, sampling_spec=sampling) == (3, 512, 768)


def test_data_time_velocity_is_negated_for_sigma_solver() -> None:
    class ConstantVelocityStep(SenseNovaU1DiffusionStep):
        def predict_velocity(self, *args, sample, **kwargs):
            return torch.full_like(sample, 2.0)

    state = torch.zeros(1, 2, 3)
    next_state, log_prob, _ = ConstantVelocityStep().step_with_logp(
        None,
        None,
        strategy=FlowSDEStrategy(),
        sample=state,
        sigma=torch.tensor(1.0),
        sigma_next=torch.tensor(0.5),
        params=None,
        eta=0.0,
    )

    # Upstream integrates dx/dt=2 over dt=0.5. The framework integrates over
    # decreasing sigma, so its noise prediction must be dx/dsigma=-2.
    torch.testing.assert_close(next_state, torch.ones_like(state))
    assert log_prob is None
