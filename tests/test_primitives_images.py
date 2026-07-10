"""Unit tests for dense outputs and native-resolution image inputs."""

from __future__ import annotations

import pytest
import torch

from unirl.types.primitives import Image, Images, NativeImages


def _rgb(h: int, w: int, fill: float = 0.5) -> Image:
    return Image(pixels=torch.full((3, h, w), fill, dtype=torch.float32))


def test_images_from_list_uniform_stacks() -> None:
    batch = Images.from_list([_rgb(64, 64, 0.25), _rgb(64, 64, 0.75)])
    assert batch.pixels.shape == (2, 3, 64, 64)
    assert torch.allclose(batch.pixels[0], torch.full((3, 64, 64), 0.25))
    assert torch.allclose(batch.pixels[1], torch.full((3, 64, 64), 0.75))


def test_images_from_list_mixed_hw_raises() -> None:
    with pytest.raises(ValueError, match="NativeImages"):
        Images.from_list([_rgb(32, 48), _rgb(64, 96)])


def test_native_images_preserve_mixed_resolution_roundtrip() -> None:
    first = _rgb(16, 24, 0.25)
    second = _rgb(32, 48, 0.75)
    batch = NativeImages.from_list([first, second])
    items = batch.to_list()

    assert len(batch) == batch.batch_size == 2
    assert len(items) == 2
    assert items[0].pixels.shape == (3, 16, 24)
    assert items[1].pixels.shape == (3, 32, 48)
    assert torch.equal(items[0].pixels, first.pixels)
    assert torch.equal(items[1].pixels, second.pixels)


def test_native_images_batch_ops_preserve_shapes_and_order() -> None:
    left = NativeImages.from_list([_rgb(8, 12, 0.1), _rgb(16, 20, 0.2)])
    right = NativeImages.from_list([_rgb(24, 28, 0.3)])

    merged = NativeImages.concat([left, right])
    assert [tuple(p.shape) for p in merged.pixels] == [(3, 8, 12), (3, 16, 20), (3, 24, 28)]
    assert torch.allclose(merged.pixels[2], torch.full((3, 24, 28), 0.3))

    selected = merged.select([2, 0])
    assert [tuple(p.shape) for p in selected.pixels] == [(3, 24, 28), (3, 8, 12)]

    sliced = merged.slice(1, 3)
    assert [tuple(p.shape) for p in sliced.pixels] == [(3, 16, 20), (3, 24, 28)]

    repeated = left.repeat_interleave(2)
    assert [tuple(p.shape) for p in repeated.pixels] == [(3, 8, 12), (3, 8, 12), (3, 16, 20), (3, 16, 20)]


def test_images_from_list_mixed_channels_raises() -> None:
    rgb = Image(pixels=torch.zeros(3, 16, 16))
    gray = Image(pixels=torch.zeros(1, 16, 16))
    with pytest.raises(ValueError, match="channel count"):
        Images.from_list([rgb, gray])
    with pytest.raises(ValueError, match="channel count"):
        NativeImages.from_list([rgb, gray])


def test_images_from_list_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        Images.from_list([])
    with pytest.raises(ValueError, match="empty"):
        NativeImages.from_list([])
