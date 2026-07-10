#!/usr/bin/env python
"""UniRL AR (autoregressive) evaluation entry point (Hydra-native).

Standalone counterpart to ``train_ar.py``: build the same
:class:`~unirl.trainer.ar.ARTrainer` from a recipe, then run a single
evaluation (``avg@k`` accuracy on the eval prompt set, scored by the configured
reward service, logged under ``eval/*``) instead of the train loop — no
``num_rollouts=0`` workaround.

Evaluate a checkpoint on a benchmark prompt set::

    # Full-weight ckpt (base model via the recipe's model path env, e.g. QWEN_VL_PATH)
    EVAL_DATA_PATH=/path/to/val.jsonl \\
    python -m unirl.eval_ar --config-name=ar/qwen_vl_grpo_geo3k_mc_4x8 \\
      +eval_num_prompts=<N> +eval_samples_per_prompt=<K>

    # Trained LoRA adapter (base stays the recipe's model path)
    ... python -m unirl.eval_ar ... +load_dir=<checkpoint_dir>

``eval_num_prompts=-1`` (default) evaluates the full validation set (#189).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.ar import ARTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="ar/qwen_vl_grpo_geo3k_mc_4x8")
def main(cfg: DictConfig) -> None:
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


if __name__ == "__main__":
    main()
