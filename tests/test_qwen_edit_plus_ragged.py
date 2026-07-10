"""Mixed-aspect Qwen Image Edit Plus condition tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest
import torch

from unirl.models.qwen_image.diffusion import _pack_latents
from unirl.models.qwen_image_edit_plus import vae as edit_vae
from unirl.models.qwen_image_edit_plus.conditions import QwenImageEditPlusConditions
from unirl.models.qwen_image_edit_plus.diffusion import QwenImageEditPlusDiffusionStep
from unirl.models.qwen_image_edit_plus.vae import QwenImageEditPlusVAEEncodeStage, _vae_size_for_aspect
from unirl.rollout.engine.sglang_diffusion._patches.patch_conditions import (
    _copy_conditions,
    _merge_conditions,
    _normalize_vae_image_sizes,
    _slice_image_size_list,
    _slice_ragged_tensor_list,
)
from unirl.rollout.engine.sglang_diffusion.adapters.qwen_image_edit_plus import QwenImageEditPlusAdapter
from unirl.types.conditions import ImageLatentCondition, RaggedImageLatentCondition, TextEmbedCondition
from unirl.types.primitives import NativeImages, Texts
from unirl.types.rollout_req import RolloutReq


class _ProbeStep(QwenImageEditPlusDiffusionStep):
    def __init__(self) -> None:
        self.calls: List[dict[str, Any]] = []

    def _predict_noise_uniform(
        self,
        model,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: QwenImageEditPlusConditions,
        image_latents: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del model, kwargs
        assert conditions.text is not None and conditions.text.embeds is not None
        markers = conditions.text.embeds[:, 0, 0]
        self.calls.append(
            {
                "markers": markers.detach().cpu().tolist(),
                "sigma": sigma.detach().cpu().tolist(),
                "latent_shape": tuple(image_latents.shape),
            }
        )
        return sample + markers.view(-1, 1, 1, 1)


def _conditions() -> QwenImageEditPlusConditions:
    markers = torch.arange(4, dtype=torch.float32).view(4, 1, 1)
    text = TextEmbedCondition(embeds=markers, attn_mask=torch.ones(4, 1, dtype=torch.long))
    latents = [
        torch.zeros(16, 8, 12),
        torch.zeros(16, 12, 8),
        torch.ones(16, 8, 12),
        torch.ones(16, 12, 8),
    ]
    return QwenImageEditPlusConditions(
        text=text,
        image_latent=RaggedImageLatentCondition(latents=latents),
    )


def test_ragged_condition_concat_and_select_preserve_spatial_shapes() -> None:
    conditions = _conditions()
    selected = conditions.select([3, 0, 2])

    assert selected.image_latent is not None
    assert [tuple(latent.shape) for latent in selected.image_latent.latents] == [
        (16, 12, 8),
        (16, 8, 12),
        (16, 8, 12),
    ]

    merged = QwenImageEditPlusConditions.concat([conditions.slice(0, 2), conditions.slice(2, 4)])
    assert merged.image_latent is not None
    assert [tuple(latent.shape) for latent in merged.image_latent.latents] == [
        (16, 8, 12),
        (16, 12, 8),
        (16, 8, 12),
        (16, 12, 8),
    ]


def test_conditions_from_dict_upgrades_legacy_dense_image_latents() -> None:
    text = TextEmbedCondition(embeds=torch.zeros(2, 1, 1), attn_mask=torch.ones(2, 1, dtype=torch.long))
    conditions = QwenImageEditPlusConditions.from_dict(
        {
            "text": text,
            "image_latent": ImageLatentCondition(latents=torch.zeros(2, 16, 8, 12)),
        }
    )

    assert conditions.image_latent is not None
    assert [tuple(latent.shape) for latent in conditions.image_latent.latents] == [
        (16, 8, 12),
        (16, 8, 12),
    ]


def test_predict_noise_shape_microbatches_restore_order_and_gradient() -> None:
    step = _ProbeStep()
    sample = torch.zeros(4, 4, 2, 2, requires_grad=True)
    sigma = torch.tensor([0.1, 0.2, 0.3, 0.4])

    output = step.predict_noise(
        object(),
        sample,
        sigma,
        _conditions(),
        guidance_scale=1.0,
        latent_h=2,
        latent_w=2,
    )

    expected = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand_as(output)
    assert torch.equal(output, expected)
    assert [call["markers"] for call in step.calls] == [[0.0, 2.0], [1.0, 3.0]]
    assert [call["latent_shape"] for call in step.calls] == [(2, 16, 8, 12), (2, 16, 12, 8)]
    assert step.calls[0]["sigma"] == pytest.approx([0.1, 0.3])
    assert step.calls[1]["sigma"] == pytest.approx([0.2, 0.4])

    output.sum().backward()
    assert torch.equal(sample.grad, torch.ones_like(sample))


def test_predict_noise_shape_microbatches_apply_cfg_with_scalar_sigma() -> None:
    class _Transformer:
        config = SimpleNamespace(guidance_embeds=False)

        def __init__(self) -> None:
            self.calls: List[dict[str, Any]] = []

        def __call__(
            self,
            *,
            hidden_states,
            timestep,
            encoder_hidden_states,
            **kwargs,
        ):
            del kwargs
            markers = encoder_hidden_states[:, 0, 0]
            self.calls.append(
                {
                    "markers": markers.detach().cpu().tolist(),
                    "timestep": timestep.detach().cpu().tolist(),
                    "hidden_shape": tuple(hidden_states.shape),
                }
            )
            return (torch.ones_like(hidden_states) * markers.view(-1, 1, 1),)

    transformer = _Transformer()
    model = SimpleNamespace(transformer=transformer)
    text = TextEmbedCondition(
        embeds=torch.full((4, 1, 1), 2.0),
        attn_mask=torch.ones(4, 1, dtype=torch.long),
    )
    negative = TextEmbedCondition(
        embeds=torch.ones(4, 1, 1),
        attn_mask=torch.ones(4, 1, dtype=torch.long),
    )
    conditions = QwenImageEditPlusConditions(
        text=text,
        negative_text=negative,
        image_latent=RaggedImageLatentCondition(
            latents=[
                torch.zeros(16, 2, 4),
                torch.zeros(16, 4, 2),
                torch.ones(16, 2, 4),
                torch.ones(16, 4, 2),
            ]
        ),
    )

    output = QwenImageEditPlusDiffusionStep().predict_noise(
        model,
        torch.zeros(4, 16, 2, 2),
        torch.tensor(0.25),
        conditions,
        guidance_scale=2.0,
        latent_h=2,
        latent_w=2,
    )

    assert output.shape == (4, 16, 2, 2)
    assert torch.allclose(output, torch.full_like(output, 2.0))
    assert [call["markers"] for call in transformer.calls] == [
        [2.0, 2.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [1.0, 1.0],
    ]
    assert all(call["timestep"] == [0.25, 0.25] for call in transformer.calls)


def test_vae_size_preserves_orientation_without_batch_coupling() -> None:
    landscape = _vae_size_for_aspect(width=400, height=100)
    portrait = _vae_size_for_aspect(width=100, height=400)

    assert landscape == tuple(reversed(portrait))
    assert landscape[0] > landscape[1]
    assert portrait[0] < portrait[1]


class _FakeLatentDist:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class _FakeVAE:
    def __init__(self) -> None:
        self.config = SimpleNamespace(z_dim=2, latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0])
        self.encode_shapes: List[tuple[int, ...]] = []

    def to(self, dtype):
        del dtype
        return self

    def encode(self, value: torch.Tensor):
        self.encode_shapes.append(tuple(value.shape))
        batch, _, _, height, width = value.shape
        latent = torch.zeros(batch, 2, 1, height // 8, width // 8)
        return SimpleNamespace(latent_dist=_FakeLatentDist(latent))


def test_vae_encode_buckets_native_images_by_aspect_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(edit_vae, "_VAE_IMAGE_AREA", 64 * 64)
    fake_vae = _FakeVAE()
    stage = QwenImageEditPlusVAEEncodeStage(
        SimpleNamespace(vae=fake_vae, device=torch.device("cpu"), dtype=torch.float32)
    )
    images = NativeImages(
        pixels=[
            torch.zeros(3, 8, 16),
            torch.zeros(3, 16, 8),
            torch.zeros(3, 4, 8),
        ]
    )

    encoded = stage.encode(images, height=32, width=32)

    assert fake_vae.encode_shapes == [(2, 3, 1, 32, 96), (1, 3, 1, 96, 32)]
    assert [tuple(latent.shape) for latent in encoded.latents] == [
        (2, 4, 12),
        (2, 12, 4),
        (2, 4, 12),
    ]


def test_sglang_condition_capture_keeps_heterogeneous_latents_ragged() -> None:
    landscape = torch.arange(16 * 4 * 12, dtype=torch.float32).view(1, 16, 4, 12)
    portrait = torch.arange(16 * 12 * 4, dtype=torch.float32).view(1, 16, 12, 4)
    results = [
        SimpleNamespace(
            image_latent=[_pack_latents(landscape)],
            image_latent_sizes=[[(96, 32)]],
        ),
        SimpleNamespace(
            image_latent=[_pack_latents(portrait)],
            image_latent_sizes=[[(32, 96)]],
        ),
    ]
    adapter = object.__new__(QwenImageEditPlusAdapter)

    captured = adapter._collect_image_latents(results)

    assert captured is not None
    assert [tuple(latent.shape) for latent in captured] == [(16, 4, 12), (16, 12, 4)]
    assert torch.equal(captured[0], landscape[0])
    assert torch.equal(captured[1], portrait[0])


def test_sglang_size_metadata_merges_and_slices_per_output() -> None:
    first = torch.zeros(1, 12, 64)
    second = torch.zeros(1, 16, 64)
    copied = SimpleNamespace()
    _copy_conditions(
        SimpleNamespace(
            return_prompt_embeds=False,
            return_negative_prompt_embeds=False,
            image_latent=[first, second],
            vae_image_sizes=[(96, 32), (32, 96)],
        ),
        copied,
    )
    assert len(copied.image_latent) == 1
    assert len(copied.image_latent[0]) == 2
    assert torch.equal(copied.image_latent[0][0], first)
    assert torch.equal(copied.image_latent[0][1], second)
    assert copied.image_latent_sizes == [[[(96, 32)], [(32, 96)]]]

    output_batches = [
        SimpleNamespace(image_latent=[[first]], image_latent_sizes=[[[(96, 32)]]]),
        SimpleNamespace(image_latent=[[second]], image_latent_sizes=[[[(32, 96)]]]),
    ]
    merged = SimpleNamespace()

    _merge_conditions(merged, output_batches)

    assert len(merged.image_latent) == 1
    assert torch.equal(merged.image_latent[0][0], first)
    assert torch.equal(merged.image_latent[0][1], second)
    assert merged.image_latent_sizes == [[[(96, 32)], [(32, 96)]]]
    assert torch.equal(_slice_ragged_tensor_list(merged.image_latent, 0)[0], first)
    assert torch.equal(_slice_ragged_tensor_list(merged.image_latent, 1)[0], second)
    assert _slice_image_size_list(merged.image_latent_sizes, 0) == [[(96, 32)]]
    assert _slice_image_size_list(merged.image_latent_sizes, 1) == [[(32, 96)]]


def test_sglang_size_metadata_normalizes_flat_and_nested_batches() -> None:
    assert _normalize_vae_image_sizes([(96, 32)], 1) == [[(96, 32)]]
    assert _normalize_vae_image_sizes([[(96, 32)]], 1) == [[(96, 32)]]
    assert _normalize_vae_image_sizes([(96, 32), (32, 96)], 2) == [[(96, 32)], [(32, 96)]]
    assert _normalize_vae_image_sizes([[(96, 32)], [(32, 96)]], 2) == [[(96, 32)], [(32, 96)]]

    with pytest.raises(ValueError, match="batch mismatch"):
        _normalize_vae_image_sizes([(96, 32)], 2)
    with pytest.raises(ValueError, match="must be"):
        _normalize_vae_image_sizes([[96, 32]], 1)
    with pytest.raises(IndexError, match="legacy unbatched"):
        _slice_image_size_list([[(96, 32), (32, 96)]], 1)


def test_edit_adapter_build_prompts_preserves_native_sizes_when_deexpanded() -> None:
    adapter = object.__new__(QwenImageEditPlusAdapter)
    req = RolloutReq(
        sample_ids=["a0", "b0", "a1", "b1"],
        group_ids=["a", "b", "a", "b"],
        primitives={
            "text": Texts(texts=["landscape", "portrait", "landscape", "portrait"]),
            "image": NativeImages(
                pixels=[
                    torch.zeros(3, 8, 16),
                    torch.zeros(3, 16, 8),
                    torch.ones(3, 8, 16),
                    torch.ones(3, 16, 8),
                ]
            ),
        },
    )

    prompts = adapter.build_prompts(req)

    assert prompts["prompt"] == ["landscape", "portrait"]
    assert prompts["num_outputs_per_prompt"] == 2
    assert [image.size for image in prompts["condition_image"]] == [(16, 8), (8, 16)]
