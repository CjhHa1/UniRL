from types import SimpleNamespace

import pytest
import torch

from unirl.rollout.engine.vllm_omni.adapters.sd3 import Sd3InputAdapter, Sd3OutputAdapter
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import _grouped_span, resolve_request_noise
from unirl.rollout.engine.vllm_omni.utils.prompts import grouped_texts_from_req
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import DiffusionSamplingParams


def _req(texts, *, spp=4, group_ids=None, init_noise_group_ids=None):
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(len(texts))],
        group_ids=group_ids or [f"g{i // spp}" for i in range(len(texts))],
        primitives={"text": Texts(texts=list(texts))},
        sampling_params={
            "diffusion": DiffusionSamplingParams(
                samples_per_prompt=spp,
                num_inference_steps=2,
                guidance_scale=1.0,
                height=512,
                width=512,
                eta=0.7,
                seed=123,
            )
        },
        init_noise_group_ids=init_noise_group_ids or [f"n{i}" for i in range(len(texts))],
        init_noise_latent_shape=[16, 64, 64],
    )


def test_grouped_texts_from_req_collapses_contiguous_prompt_groups():
    req = _req(["cat"] * 4 + ["dog"] * 4, spp=4)

    grouped, spp = grouped_texts_from_req(req, samples_per_prompt=4, caller="test")

    assert grouped == ["cat", "dog"]
    assert spp == 4


def test_grouped_texts_from_req_rejects_split_or_interleaved_groups():
    req = _req(["cat", "cat", "dog", "cat"], spp=4)

    with pytest.raises(RuntimeError, match="not contiguous/repeated"):
        grouped_texts_from_req(req, samples_per_prompt=4, caller="test")


def test_sd3_input_adapter_uses_num_outputs_per_prompt_and_sample_level_noise_ids():
    req = _req(["cat"] * 4 + ["dog"] * 4, spp=4)
    adapter = Sd3InputAdapter("sd3_t2i")

    prompts = adapter.build_prompts(req)
    sampling = adapter.build_sampling(req)

    assert [p["prompt"] for p in prompts] == ["cat", "dog"]
    assert sampling[0].kwargs["num_outputs_per_prompt"] == 4
    assert sampling[0].kwargs["extra_args"]["init_noise_group_ids"] == [f"n{i}" for i in range(8)]


def test_grouped_noise_tensor_slice_uses_num_outputs_per_prompt_span():
    noise = torch.arange(8 * 2, dtype=torch.float32).view(8, 2)
    req = SimpleNamespace(
        request_id="1_abcd",
        sampling_params=SimpleNamespace(num_outputs_per_prompt=4, extra_args={"initial_noise_batch": noise}),
    )

    out = resolve_request_noise(req, caller="test")

    assert torch.equal(out, noise[4:8])
    assert _grouped_span(1, 4) == (4, 8)


def test_sd3_output_adapter_repeats_prompt_level_text_capture_to_sample_level():
    req = _req(["cat"] * 4 + ["dog"] * 4, spp=4)
    diff_outputs = [
        SimpleNamespace(
            final_output_type="image",
            stage_id=0,
            images=[object()] * 4,
            trajectory_latents=torch.randn(4, 3, 2, 2),
            custom_output={
                "text_capture": {
                    "prompt_embeds": torch.full((1, 2, 3), float(i)),
                    "pooled_prompt_embeds": torch.full((1, 3), float(i)),
                }
            },
        )
        for i in range(2)
    ]
    per_request = [[out] for out in diff_outputs]

    conds = Sd3OutputAdapter("sd3_t2i").build_conditions(req, per_request)
    text = conds["text"]

    assert text.embeds.shape == (8, 2, 3)
    assert text.pooled.shape == (8, 3)
    assert torch.equal(text.embeds[:4], torch.zeros(4, 2, 3))
    assert torch.equal(text.embeds[4:], torch.ones(4, 2, 3))


def test_sd3_output_adapter_rejects_misaligned_condition_batch():
    req = _req(["cat"] * 4 + ["dog"] * 4, spp=4)
    diff_outputs = [
        SimpleNamespace(
            final_output_type="image",
            stage_id=0,
            images=[object()] * 3,
            trajectory_latents=torch.randn(3, 3, 2, 2),
            custom_output={
                "text_capture": {
                    "prompt_embeds": torch.zeros(1, 2, 3),
                    "pooled_prompt_embeds": torch.zeros(1, 3),
                }
            },
        ),
        SimpleNamespace(
            final_output_type="image",
            stage_id=0,
            images=[object()] * 4,
            trajectory_latents=torch.randn(4, 3, 2, 2),
            custom_output={
                "text_capture": {
                    "prompt_embeds": torch.ones(1, 2, 3),
                    "pooled_prompt_embeds": torch.ones(1, 3),
                }
            },
        ),
    ]

    with pytest.raises(RuntimeError, match="does not divide"):
        Sd3OutputAdapter("sd3_t2i").build_conditions(req, [[out] for out in diff_outputs])
