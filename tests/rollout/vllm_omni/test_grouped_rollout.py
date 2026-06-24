"""Unit tests for grouped prompt rollout — validates SD3 + Qwen-Image adapters.

Tests the core grouping helper (``grouped_texts_from_req``), the SD3 and
Qwen-Image input adapters' grouped prompt/sampling construction, the
interception layer's ``_grouped_span`` noise slicing, and the SD3 output
adapter's condition repeat_interleave logic.

Run: ``python -m pytest tests/rollout/vllm_omni/test_grouped_rollout.py -v``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch

from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import _grouped_span
from unirl.rollout.engine.vllm_omni.utils.prompts import grouped_texts_from_req

# ---------------------------------------------------------------------------
# Minimal mocks
# ---------------------------------------------------------------------------


@dataclass
class FakeTexts:
    texts: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.texts)


@dataclass
class FakeDiffParams:
    samples_per_prompt: int = 1
    negative_prompt: str = ""
    guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    num_inference_steps: int = 28
    eta: float = 0.0
    seed: Optional[int] = None
    max_sequence_length: Optional[int] = None
    sde_indices: Optional[List[int]] = None


@dataclass
class FakeReq:
    sample_ids: List[str] = field(default_factory=list)
    group_ids: List[str] = field(default_factory=list)
    primitives: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    sigmas: Optional[Any] = None
    init_noise_group_ids: List[str] = field(default_factory=list)
    init_noise_latent_shape: Optional[List[int]] = None
    request_conditions: Optional[Dict[str, Any]] = None
    stage_config: Optional[Dict[str, Any]] = None

    def get_sigmas(self):
        return self.sigmas


def make_req(n_samples: int, spp: int, *, guidance: float = 1.0) -> FakeReq:
    """Build a FakeReq with n_samples, grouped by spp."""
    n_prompts = n_samples // spp
    texts = []
    sample_ids = []
    group_ids = []
    for p in range(n_prompts):
        for s in range(spp):
            texts.append(f"prompt_{p}")
            sample_ids.append(f"s{p}_{s}")
            group_ids.append(f"g{p}")
    return FakeReq(
        sample_ids=sample_ids,
        group_ids=group_ids,
        primitives={"text": FakeTexts(texts=texts)},
        sampling_params={
            "diffusion": FakeDiffParams(
                samples_per_prompt=spp,
                guidance_scale=guidance,
            )
        },
    )


# ---------------------------------------------------------------------------
# Tests: grouped_texts_from_req
# ---------------------------------------------------------------------------


class TestGroupedTextsFromReq:
    def test_spp1_passthrough(self):
        req = make_req(4, spp=1)
        grouped, spp_out = grouped_texts_from_req(req, samples_per_prompt=1, caller="test")
        assert spp_out == 1
        assert grouped == ["prompt_0", "prompt_1", "prompt_2", "prompt_3"]

    def test_spp2_groups(self):
        req = make_req(4, spp=2)
        grouped, spp_out = grouped_texts_from_req(req, samples_per_prompt=2, caller="test")
        assert spp_out == 2
        assert grouped == ["prompt_0", "prompt_1"]

    def test_spp4_single_group(self):
        req = make_req(4, spp=4)
        grouped, spp_out = grouped_texts_from_req(req, samples_per_prompt=4, caller="test")
        assert spp_out == 4
        assert grouped == ["prompt_0"]

    def test_indivisible_raises(self):
        # 5 samples, trying to group by spp=2 → not divisible
        req = FakeReq(
            sample_ids=[f"s{i}" for i in range(5)],
            group_ids=[f"g{i}" for i in range(5)],
            primitives={"text": FakeTexts(texts=[f"p{i}" for i in range(5)])},
            sampling_params={"diffusion": FakeDiffParams(samples_per_prompt=2)},
        )
        with pytest.raises(RuntimeError, match="not divisible"):
            grouped_texts_from_req(req, samples_per_prompt=2, caller="test")

    def test_non_contiguous_texts_raises(self):
        req = make_req(4, spp=2)
        req.primitives["text"].texts[1] = "different_prompt"
        with pytest.raises(RuntimeError, match="not contiguous"):
            grouped_texts_from_req(req, samples_per_prompt=2, caller="test")

    def test_non_contiguous_group_ids_raises(self):
        req = make_req(4, spp=2)
        req.group_ids[1] = "different_group"
        with pytest.raises(RuntimeError, match="group_ids.*not contiguous"):
            grouped_texts_from_req(req, samples_per_prompt=2, caller="test")


# ---------------------------------------------------------------------------
# Tests: _grouped_span
# ---------------------------------------------------------------------------


class TestGroupedSpan:
    def test_spp1(self):
        assert _grouped_span(0, 1) == (0, 1)
        assert _grouped_span(3, 1) == (3, 4)

    def test_spp4(self):
        assert _grouped_span(0, 4) == (0, 4)
        assert _grouped_span(1, 4) == (4, 8)
        assert _grouped_span(2, 4) == (8, 12)

    def test_invalid_spp_raises(self):
        with pytest.raises(ValueError):
            _grouped_span(0, -1)

    def test_spp_zero_defaults_to_one(self):
        assert _grouped_span(2, 0) == (2, 3)


# ---------------------------------------------------------------------------
# Tests: Qwen-Image grouped input adapter
# ---------------------------------------------------------------------------


class TestQwenImageGroupedInputAdapter:
    def _make_adapter(self):
        from unirl.rollout.engine.vllm_omni.adapters.qwen_image import QwenImageGroupedInputAdapter

        model_config = MagicMock()
        model_config.max_sequence_length = 512
        return QwenImageGroupedInputAdapter("qwen_image_t2i", model_config=model_config)

    def test_build_prompts_spp2_no_cfg(self):
        adapter = self._make_adapter()
        req = make_req(4, spp=2, guidance=1.0)
        prompts = adapter.build_prompts(req)
        assert len(prompts) == 2
        assert prompts[0] == {"prompt": "prompt_0"}
        assert prompts[1] == {"prompt": "prompt_1"}
        assert "negative_prompt" not in prompts[0]

    def test_build_prompts_spp2_with_cfg(self):
        adapter = self._make_adapter()
        req = make_req(4, spp=2, guidance=4.5)
        req.sampling_params["diffusion"].negative_prompt = "bad quality"
        prompts = adapter.build_prompts(req)
        assert len(prompts) == 2
        assert prompts[0] == {"prompt": "prompt_0", "negative_prompt": "bad quality"}

    def test_build_sampling_sets_num_outputs(self):
        adapter = self._make_adapter()
        req = make_req(4, spp=2, guidance=1.0)
        req.sigmas = None
        sampling = adapter.build_sampling(req)
        assert sampling[0].kwargs["num_outputs_per_prompt"] == 2

    def test_build_sampling_spp1_keeps_default(self):
        adapter = self._make_adapter()
        req = make_req(4, spp=1, guidance=1.0)
        req.sigmas = None
        sampling = adapter.build_sampling(req)
        assert sampling[0].kwargs["num_outputs_per_prompt"] == 1


# ---------------------------------------------------------------------------
# Tests: SD3 grouped input adapter
# ---------------------------------------------------------------------------


class TestSd3InputAdapter:
    def _make_adapter(self):
        from unirl.rollout.engine.vllm_omni.adapters.sd3 import Sd3InputAdapter

        return Sd3InputAdapter("sd3_t2i")

    def test_build_prompts_grouped(self):
        adapter = self._make_adapter()
        req = make_req(6, spp=3)
        req.sampling_params["diffusion"].negative_prompt = ""
        prompts = adapter.build_prompts(req)
        assert len(prompts) == 2
        assert prompts[0]["prompt"] == "prompt_0"
        assert prompts[1]["prompt"] == "prompt_1"

    def test_build_sampling_num_outputs(self):
        adapter = self._make_adapter()
        req = make_req(6, spp=3)
        req.sigmas = None
        sampling = adapter.build_sampling(req)
        assert sampling[0].kwargs["num_outputs_per_prompt"] == 3


# ---------------------------------------------------------------------------
# Tests: SD3 output adapter condition repeat
# ---------------------------------------------------------------------------


class TestSd3OutputAdapterConditions:
    def _make_adapter(self):
        from unirl.rollout.engine.vllm_omni.adapters.sd3 import Sd3OutputAdapter

        return Sd3OutputAdapter("sd3_t2i")

    def _make_diff_output(self, batch_size: int, embed_dim: int = 64, seq_len: int = 77):
        out = MagicMock()
        out.custom_output = {
            "text_capture": {
                "prompt_embeds": torch.randn(1, seq_len, embed_dim),
                "pooled_prompt_embeds": torch.randn(1, embed_dim),
            }
        }
        out.trajectory_latents = torch.randn(batch_size, 10, 4, 8, 8)
        out.stage_id = 0
        out.final_output_type = "image"
        # Provide fake PIL images so collect_dit_outputs doesn't complain
        out.images = [MagicMock() for _ in range(batch_size)]
        return out

    def test_repeat_interleave_spp2(self):
        adapter = self._make_adapter()
        # 2 requests, each with trajectory batch=2 (spp=2)
        out0 = self._make_diff_output(batch_size=2)
        out1 = self._make_diff_output(batch_size=2)
        per_request = [[out0], [out1]]
        req = make_req(4, spp=2)
        conds = adapter.build_conditions(req, per_request)
        assert conds["text"].embeds.shape[0] == 4
        assert conds["text"].pooled.shape[0] == 4

    def test_no_repeat_spp1(self):
        adapter = self._make_adapter()
        out0 = self._make_diff_output(batch_size=1)
        out1 = self._make_diff_output(batch_size=1)
        per_request = [[out0], [out1]]
        req = make_req(2, spp=1)
        conds = adapter.build_conditions(req, per_request)
        assert conds["text"].embeds.shape[0] == 2
        assert conds["text"].pooled.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
