"""Unit tests for Images.from_list mixed-resolution handling (issue #169)."""

from __future__ import annotations

import logging

import pytest
import torch

from unirl.types.primitives import Image, Images


def _rgb(h: int, w: int, fill: float = 0.5) -> Image:
    return Image(pixels=torch.full((3, h, w), fill, dtype=torch.float32))


def test_images_from_list_uniform_stacks() -> None:
    batch = Images.from_list([_rgb(64, 64, 0.25), _rgb(64, 64, 0.75)])
    assert batch.pixels.shape == (2, 3, 64, 64)
    assert torch.allclose(batch.pixels[0], torch.full((3, 64, 64), 0.25))
    assert torch.allclose(batch.pixels[1], torch.full((3, 64, 64), 0.75))


def test_images_from_list_mixed_resizes_not_pads(caplog: pytest.LogCaptureFixture) -> None:
    # Smaller image filled with 1.0 — zero-pad would leave a black (0) border;
    # bilinear resize to batch-max must not introduce a zero corner strip.
    small = Image(pixels=torch.ones(3, 32, 48, dtype=torch.float32))
    large = Image(pixels=torch.full((3, 64, 96), 0.5, dtype=torch.float32))
    with caplog.at_level(logging.WARNING, logger="unirl.types.primitives"):
        batch = Images.from_list([small, large])
    assert batch.pixels.shape == (2, 3, 64, 96)
    # No zero-pad border: every pixel of the resized small image stays near 1.0.
    assert float(batch.pixels[0].min()) > 0.9
    assert "mixed H/W" in caplog.text


def test_images_to_list_returns_resized_tensors() -> None:
    batch = Images.from_list([_rgb(16, 24), _rgb(32, 48)])
    items = batch.to_list()
    assert len(items) == 2
    assert items[0].pixels.shape == (3, 32, 48)
    assert items[1].pixels.shape == (3, 32, 48)


def test_images_from_list_mixed_channels_raises() -> None:
    rgb = Image(pixels=torch.zeros(3, 16, 16))
    gray = Image(pixels=torch.zeros(1, 16, 16))
    with pytest.raises(ValueError, match="channel count"):
        Images.from_list([rgb, gray])


def test_images_from_list_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        Images.from_list([])
