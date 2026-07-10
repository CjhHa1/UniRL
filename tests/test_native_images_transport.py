"""Transport and DP regression tests for native-resolution image lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from unirl.distributed.tensor.pytree import pytree_cat, pytree_chunk
from unirl.distributed.tensor.ref import TensorRef
from unirl.distributed.tensor.transport import TensorTransport
from unirl.types.conditions import RaggedImageLatentCondition
from unirl.types.primitives import Image, Images, NativeImages


@dataclass
class _Handle:
    tensor: torch.Tensor

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def device(self):
        return self.tensor.device

    def local(self) -> torch.Tensor:
        return self.tensor


class _MemoryTransport(TensorTransport):
    def put(self, tensor: torch.Tensor) -> _Handle:
        return _Handle(tensor.detach().clone())

    def _resolve_handles(self, handles: List[_Handle]) -> List[torch.Tensor]:
        return [handle.tensor for handle in handles]

    def is_ref(self, value) -> bool:
        return isinstance(value, TensorRef)


def _native_batch() -> NativeImages:
    shapes = [(3, 8, 12), (3, 16, 20), (3, 24, 28), (3, 32, 36)]
    return NativeImages.from_list(
        [Image(pixels=torch.full(shape, float(index + 1))) for index, shape in enumerate(shapes)]
    )


def test_native_images_transport_roundtrip_preserves_each_chw_tensor() -> None:
    backend = _MemoryTransport()
    original = _native_batch()
    dehydrated = backend.dehydrate(original.clone())

    assert len(dehydrated.pixels) == 4
    assert all(isinstance(value, TensorRef) for value in dehydrated.pixels)
    # Each ref spans the complete CHW tensor. Its row metadata counts channels;
    # the surrounding list—not the inner ref—is the sample axis.
    assert [value.batch_size for value in dehydrated.pixels] == [3, 3, 3, 3]

    hydrated = backend.hydrate(dehydrated)
    assert [tuple(value.shape) for value in hydrated.pixels] == [
        (3, 8, 12),
        (3, 16, 20),
        (3, 24, 28),
        (3, 32, 36),
    ]
    for actual, expected in zip(hydrated.pixels, original.pixels):
        assert torch.equal(actual, expected)


def test_dehydrated_native_images_dp_chunk_cat_roundtrip() -> None:
    backend = _MemoryTransport()
    original = _native_batch()
    dehydrated = backend.dehydrate(original.clone())

    shards = pytree_chunk(dehydrated, dp_size=2, batch_size=4)
    assert [len(shard) for shard in shards] == [2, 2]
    merged = pytree_cat(shards)
    hydrated = backend.hydrate(merged)

    assert len(hydrated) == 4
    for actual, expected in zip(hydrated.pixels, original.pixels):
        assert torch.equal(actual, expected)


def test_ragged_image_latent_condition_transport_and_dp_roundtrip() -> None:
    backend = _MemoryTransport()
    original = RaggedImageLatentCondition(
        latents=[
            torch.full((16, 4, 12), 1.0),
            torch.full((16, 12, 4), 2.0),
            torch.full((16, 4, 12), 3.0),
            torch.full((16, 12, 4), 4.0),
        ]
    )
    dehydrated = backend.dehydrate(original.clone())
    assert all(isinstance(value, TensorRef) for value in dehydrated.latents)

    merged = pytree_cat(pytree_chunk(dehydrated, dp_size=2, batch_size=4))
    hydrated = backend.hydrate(merged)

    assert [tuple(value.shape) for value in hydrated.latents] == [
        (16, 4, 12),
        (16, 12, 4),
        (16, 4, 12),
        (16, 12, 4),
    ]
    for actual, expected in zip(hydrated.latents, original.latents):
        assert torch.equal(actual, expected)


def test_dense_images_transport_contract_remains_batched() -> None:
    backend = _MemoryTransport()
    original = Images(pixels=torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5))

    dehydrated = backend.dehydrate(original.clone())

    assert isinstance(dehydrated.pixels, TensorRef)
    assert dehydrated.pixels.batch_size == 2
    hydrated = backend.hydrate(dehydrated)
    assert torch.equal(hydrated.pixels, original.pixels)
