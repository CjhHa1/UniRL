"""Integration tests for native condition-image ingress and presentation."""

from __future__ import annotations

import torch
from PIL import Image as PILImage

from unirl.data.data_source import MultimodalRLDataSource
from unirl.rollout.engine.sglang.adapters.vlm import VLMAdapter
from unirl.rollout.engine.vllm_omni.utils.prompts import pil_images_from_req
from unirl.types.media import MediaRef
from unirl.types.media_preview import build_media_preview_for_track
from unirl.types.primitives import Images, NativeImages, Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutTrack


def test_data_source_collate_preserves_native_image_sizes(tmp_path) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    PILImage.new("RGB", (12, 8), color=(64, 32, 16)).save(first_path)
    PILImage.new("RGB", (20, 16), color=(16, 32, 64)).save(second_path)

    source = object.__new__(MultimodalRLDataSource)
    inputs = source._collate_text(
        [
            {
                "prompt": "first",
                "prompt_id": "first",
                "media_refs": [MediaRef(modality="image", role="condition", uri=str(first_path))],
            },
            {
                "prompt": "second",
                "prompt_id": "second",
                "media_refs": [MediaRef(modality="image", role="condition", uri=str(second_path))],
            },
        ]
    )

    images = inputs.primitives["image"]
    assert isinstance(images, NativeImages)
    assert [tuple(pixels.shape) for pixels in images.pixels] == [(3, 8, 12), (3, 16, 20)]

    pils = pil_images_from_req(inputs, 2)
    assert [pil.size for pil in pils] == [(12, 8), (20, 16)]


def test_media_preview_pairs_native_inputs_with_dense_outputs() -> None:
    native = NativeImages(
        pixels=[
            torch.full((3, 8, 12), 0.25),
            torch.full((3, 16, 20), 0.75),
        ]
    )
    req = RolloutReq(
        sample_ids=["sample-0", "sample-1"],
        group_ids=["group-0", "group-1"],
        primitives={"image": native},
    )
    track = RolloutTrack(
        sample_ids=["sample-0", "sample-1"],
        parent_ids=["group-0", "group-1"],
        decoded=Images(pixels=torch.zeros(2, 3, 20, 20)),
    )

    preview = build_media_preview_for_track(req=req, track=track, max_items=2, prompts=["first", "second"])

    assert preview is not None
    assert [image.size for image in preview.images] == [(20, 8), (36, 16)]
    assert preview.prompts == ["first", "second"]


def test_rollout_inputs_expand_repeats_native_images_on_sample_axis() -> None:
    inputs = RolloutInputs(
        primitives={
            "text": Texts(texts=["first", "second"]),
            "image": NativeImages(
                pixels=[
                    torch.full((3, 8, 12), 0.25),
                    torch.full((3, 16, 20), 0.75),
                ]
            ),
        },
        sample_ids=["first:0", "second:0"],
        group_ids=["first", "second"],
        metadata=[None, None],
    )

    expanded = inputs.expand(2)

    images = expanded.primitives["image"]
    assert isinstance(images, NativeImages)
    assert [tuple(image.shape) for image in images.pixels] == [
        (3, 8, 12),
        (3, 8, 12),
        (3, 16, 20),
        (3, 16, 20),
    ]
    assert expanded.sample_ids == [
        "prompt:first:sample:0",
        "prompt:first:sample:1",
        "prompt:second:sample:0",
        "prompt:second:sample:1",
    ]


def test_vlm_adapter_extracts_native_pils_without_rectangularizing() -> None:
    adapter = object.__new__(VLMAdapter)
    req = RolloutReq(
        sample_ids=["sample-0", "sample-1"],
        group_ids=["group-0", "group-1"],
        primitives={
            "image": NativeImages(
                pixels=[
                    torch.zeros(3, 8, 12),
                    torch.ones(3, 16, 20),
                ]
            )
        },
    )

    images = adapter.extract_images(req, n_prompts=2)

    assert [image.size for image in images] == [(12, 8), (20, 16)]


def test_pil_images_from_req_keeps_uniform_dense_compatibility() -> None:
    req = RolloutReq(
        sample_ids=["sample-0", "sample-1"],
        group_ids=["group-0", "group-1"],
        primitives={"image": Images(pixels=torch.zeros(2, 3, 8, 12))},
    )

    images = pil_images_from_req(req, 2)

    assert [image.size for image in images] == [(12, 8), (12, 8)]
