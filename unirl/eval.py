#!/usr/bin/env python
"""Unified UniRL checkpoint evaluation entrypoint.

Select the trainer with ``--domain``; all remaining arguments are forwarded to
Hydra unchanged::

    python -m unirl.eval --domain ar \
      --config-name=ar/qwen_vl_grpo_geo3k_mc_4x8 +load_dir=<checkpoint_dir>

    python -m unirl.eval --domain diffusion \
      --config-name=diffusion/sd3/sd3_flowdppo +load_dir=<checkpoint_dir>

Each path builds the same trainer as its training counterpart and calls
``BaseTrainer.run_eval()`` once without entering the training loop.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../examples", config_name="ar/qwen_vl_grpo_geo3k_mc_4x8")
def _eval_ar(cfg: DictConfig) -> None:
    from unirl.trainer.ar import ARTrainer

    trainer = ARTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        adv_normalization_scope=cfg.get("adv_normalization_scope", "group"),
        normalize_adv_by_std=cfg.get("normalize_adv_by_std", True),
        balance_shards=cfg.get("balance_shards", False),
        eval_num_prompts=cfg.get("eval_num_prompts", -1),
        eval_batch_size=cfg.get("eval_batch_size", 8),
        eval_samples_per_prompt=cfg.get("eval_samples_per_prompt", 16),
        eval_temperature=cfg.get("eval_temperature", 1.0),
    )
    trainer.run_eval(load_dir=cfg.get("load_dir"))


@hydra.main(version_base=None, config_path="../examples", config_name="diffusion/sd3/sd3_flowdppo")
def _eval_diffusion(cfg: DictConfig) -> None:
    from unirl.trainer.diffusion import DiffusionTrainer

    trainer = DiffusionTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        layout=cfg.get("layout", "colocate"),
        train_fraction=cfg.get("train_fraction", 0.5),
        reward_fraction=cfg.get("reward_fraction", 0.0),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", False),
        adv_use_global_std=cfg.get("adv_use_global_std", False),
        eval_num_prompts=cfg.get("eval_num_prompts", 64),
        eval_samples_per_prompt=cfg.get("eval_samples_per_prompt", 4),
        eval_chunk_prompts=cfg.get("eval_chunk_prompts", 16),
        eval_cfg_text_scale=cfg.get("eval_cfg_text_scale", 4.0),
        eval_eta=cfg.get("eval_eta", 0.0),
        stage_config=cfg.get("stage_config"),
    )
    trainer.run_eval(load_dir=cfg.get("load_dir"))


def _parse_domain(argv: Sequence[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--domain", choices=("ar", "diffusion"), required=True)
    args, hydra_args = parser.parse_known_args(argv)
    return args.domain, hydra_args


def main(argv: Sequence[str] | None = None) -> None:
    domain, hydra_args = _parse_domain(sys.argv[1:] if argv is None else argv)
    entrypoints: dict[str, Callable[[], None]] = {
        "ar": _eval_ar,
        "diffusion": _eval_diffusion,
    }
    sys.argv = [sys.argv[0], *hydra_args]
    entrypoints[domain]()


if __name__ == "__main__":
    main()
