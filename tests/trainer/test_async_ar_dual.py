from omegaconf import OmegaConf

from unirl.trainer.ar import _allowed_ar_input_primitives
from unirl.trainer.async_ar import AsyncARTrainer


def test_async_ar_inherits_ar_prompt_primitive_contract() -> None:
    standard = OmegaConf.create({"_target_": "unirl.models.qwen3.pipeline.Qwen3Pipeline"})
    omni = OmegaConf.create({"_target_": "unirl.models.qwen3_omni.pipeline.Qwen3OmniPipeline"})

    assert _allowed_ar_input_primitives(standard) == {"text", "image", "video"}
    assert _allowed_ar_input_primitives(omni) == {"text", "image", "video", "media"}


def test_dual_boundary_eval_does_not_bypass_producer_publication() -> None:
    trainer = AsyncARTrainer.__new__(AsyncARTrainer)
    trainer._async_control_mode = "dual"
    calls = []
    trainer._sync_rollout = lambda **kwargs: calls.append(kwargs)

    assert trainer._prepare_rollout(sync_weights=True) is False
    assert calls == []


def test_unified_boundary_eval_keeps_manager_sync() -> None:
    trainer = AsyncARTrainer.__new__(AsyncARTrainer)
    trainer._async_control_mode = "unified"
    calls = []
    trainer._sync_rollout = lambda **kwargs: calls.append(kwargs)

    assert trainer._prepare_rollout(sync_weights=True) is False
    assert calls == [{"require_empty": True}]
