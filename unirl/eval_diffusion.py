#!/usr/bin/env python
"""UniRL diffusion evaluation entry point (Hydra-native).

Standalone counterpart to ``train_diffusion.py``: build the same
:class:`~unirl.trainer.diffusion.DiffusionTrainer` from a recipe, then run a
single evaluation (generate at the deterministic best-quality setting, score
with the configured reward service, log the mean under ``eval/*``) instead of
the train loop — no ``num_rollouts=0`` workaround.

Evaluate a checkpoint on a benchmark prompt set::

    # Full-weight ckpt (base model via the bundle's PRETRAINED_MODEL)
    PRETRAINED_MODEL=<hf-or-local-ckpt> \\
    python -m unirl.eval_diffusion --config-name=diffusion/sd3/sd3_flowdppo \\
      +eval_num_prompts=<N> +eval_samples_per_prompt=<K> \\
      data_source.args.run.eval_data_path=<benchmark_prompts.txt>

    # Trained LoRA adapter (base stays PRETRAINED_MODEL)
    ... python -m unirl.eval_diffusion ... +load_dir=<checkpoint_dir>

In-domain vs out-of-domain is just a different ``eval_data_path`` (swap the
``reward`` scorer for a different metric — GenEval2 / PickScore / HPSv3).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.diffusion import DiffusionTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="diffusion/sd3/sd3_flowdppo")
def main(cfg: DictConfig) -> None:
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


if __name__ == "__main__":
    main()
