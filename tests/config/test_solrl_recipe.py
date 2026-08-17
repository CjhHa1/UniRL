from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from unirl.trainer.diffusion import build_eval_sampling
from unirl.types.sampling import DiffusionSamplingParams


def test_solrl_h20_recipe_composes_with_expected_geometry() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples"
    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(config_name="diffusion/sd3/sd3_solrl_fp8_h20")

    assert cfg.num_devices == 32
    assert cfg.batch_size == 48
    assert cfg.scout_sampling.samples_per_prompt == 128
    assert cfg.sampling.samples_per_prompt == 16
    assert cfg.contrastive_rollout.top_k == 8
    assert cfg.contrastive_rollout.bottom_k == 8
    assert cfg.eval_sampling.num_inference_steps == 40
    assert cfg.algorithm.kl_coef == 1.0e-4


def test_paper_naive_override_arm_composes() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples"
    overrides = [
        "contrastive_rollout.mode=naive",
        "contrastive_rollout.top_k=12",
        "contrastive_rollout.bottom_k=12",
        "sampling.samples_per_prompt=24",
        "scout_sampling.samples_per_prompt=96",
        "scout_sampling.num_inference_steps=10",
        "scout_sampling.rollout_precision=bf16",
        "scout_sampling.reward_image_size=null",
        "rollout.config.fp8_enabled=false",
    ]
    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(
            config_name="diffusion/sd3/sd3_solrl_fp8_h20",
            overrides=overrides,
        )
    assert cfg.contrastive_rollout.mode == "naive"
    assert cfg.sampling.samples_per_prompt == 24
    assert cfg.scout_sampling.samples_per_prompt == 96
    assert cfg.scout_sampling.num_inference_steps == 10


def test_eval_fanout_override_keeps_deprecated_alias_in_sync() -> None:
    sampling = {
        "diffusion": DiffusionSamplingParams(
            samples_per_prompt=16,
            num_samples_per_prompt=16,
        )
    }
    evaluated = build_eval_sampling(sampling, samples_per_prompt=1)
    assert evaluated["diffusion"].samples_per_prompt == 1
    assert evaluated["diffusion"].num_samples_per_prompt == 1
