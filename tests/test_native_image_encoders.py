"""Model-boundary resize tests for native image inputs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.models.flux2_klein import vae as flux_vae
from unirl.models.flux2_klein.vae import Flux2KleinVAEEncodeStage
from unirl.models.hunyuan_image3.vit_encode import HunyuanImage3VitEncodeStage
from unirl.models.wan21.image_encode import WAN21ImageLatentEncodeStage
from unirl.types.conditions import ImageEmbedCondition
from unirl.types.primitives import NativeImages


class _LatentDist:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class _WanVAE:
    dtype = torch.float32

    def __init__(self) -> None:
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.input_shape = None

    def encode(self, value: torch.Tensor):
        self.input_shape = tuple(value.shape)
        batch, _, _, height, width = value.shape
        latent = torch.zeros(batch, 2, 2, height // 8, width // 8)
        return SimpleNamespace(latent_dist=_LatentDist(latent))


def test_wan_resizes_each_native_image_once_at_encoder_target() -> None:
    vae = _WanVAE()
    stage = WAN21ImageLatentEncodeStage(
        SimpleNamespace(vae=vae, device=torch.device("cpu"), dtype=torch.float32),
        num_frames=5,
        height=32,
        width=48,
    )
    images = NativeImages(
        pixels=[
            torch.zeros(3, 8, 24),
            torch.ones(3, 24, 8),
        ]
    )

    condition = stage.encode(images)

    assert vae.input_shape == (2, 3, 5, 32, 48)
    assert condition.latents is not None
    assert condition.latents.shape == (2, 6, 2, 4, 6)


class _FluxVAE:
    def __init__(self) -> None:
        self.input_shape = None

    def to(self, dtype):
        del dtype
        return self

    def encode(self, value: torch.Tensor):
        self.input_shape = tuple(value.shape)
        batch, _, height, width = value.shape
        latent = torch.zeros(batch, 32, height // 8, width // 8)
        return SimpleNamespace(latent_dist=_LatentDist(latent))


def test_flux_resizes_native_images_at_generation_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = _FluxVAE()
    monkeypatch.setattr(flux_vae, "normalize_patchified_latents", lambda latents, _vae: latents)
    stage = Flux2KleinVAEEncodeStage(SimpleNamespace(vae=vae, device=torch.device("cpu"), dtype=torch.float32))
    images = NativeImages(
        pixels=[
            torch.zeros(3, 8, 24),
            torch.ones(3, 24, 8),
        ]
    )

    tokens, image_ids = stage.encode(images, height=32, width=48)

    assert vae.input_shape == (2, 3, 32, 48)
    assert tokens.shape == (2, 6, 128)
    assert image_ids.shape == (2, 6, 4)


def test_hi3_cond_processor_receives_each_native_image_size() -> None:
    class _ImageProcessor:
        def __init__(self) -> None:
            self.sizes = []

        def preprocess(self, image):
            self.sizes.append(image.size)
            width, height = image.size
            num_patches = width // 4 + height // 4
            return SimpleNamespace(
                vision_image_info=SimpleNamespace(image_tensor=torch.zeros(1, num_patches, 8)),
                vision_encoder_kwargs={
                    "spatial_shapes": torch.tensor([height // 4, width // 4]),
                    "pixel_attention_mask": torch.ones(num_patches, dtype=torch.long),
                },
            )

    processor = _ImageProcessor()
    stage = HunyuanImage3VitEncodeStage(SimpleNamespace(transformer=SimpleNamespace(image_processor=processor)))
    images = NativeImages(
        pixels=[
            torch.zeros(3, 8, 16),
            torch.ones(3, 16, 8),
        ]
    )

    encoded = stage.encode_for_cond_vit(images)

    assert processor.sizes == [(16, 8), (8, 16)]
    assert [tuple(tensor.shape) for tensor in encoded["cond_vit_images"]] == [(1, 6, 8), (1, 6, 8)]
    assert [shape.tolist() for shape in encoded["vit_kwargs"]["spatial_shapes"]] == [[[2, 4]], [[4, 2]]]

    condition = ImageEmbedCondition(
        embeds=encoded["cond_vit_images"],
        attn_mask=encoded["vit_kwargs"]["attention_mask"],
        spatial_shapes=encoded["vit_kwargs"]["spatial_shapes"],
    )
    selected = condition.select([1])
    assert selected.spatial_shapes[0].tolist() == [[4, 2]]
    assert tuple(selected.embeds[0].shape) == (1, 6, 8)
