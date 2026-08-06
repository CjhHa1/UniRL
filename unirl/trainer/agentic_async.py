"""Disaggregated agentic RL over a continuously progressing rollout manager."""

from __future__ import annotations

import inspect
import logging
import sys
import time
from typing import Dict, List, Literal, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.rollout.manager import RolloutUnderflow, chain, drop_incomplete, identity, keep_within_lag
from unirl.train.stack import TrainStepResult
from unirl.trainer.agentic import AgenticTrainer
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class AsyncAgenticTrainer(AgenticTrainer):
    """Disaggregated fully-async agentic trainer (two slabs, resident engine, NCCL sync)."""

    _MAX_REFILLS = 64  # underflow guard: refills of a drained-but-short buffer before we give up

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        samples_per_prompt: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        stop: Optional[List[str]] = None,
        train_fraction: float = 0.5,
        oversample_batch_size: Optional[int] = None,
        buffer_max_staleness: Optional[int] = None,
        tail_policy: Literal["carry", "drop"] = "carry",
        per_worker_inflight: int = 8,
    ) -> None:
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        self.batch_size = int(batch_size)
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.balance_shards = False
        self.eval_interval = 0
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self.weight_sync = None
        self._stop = list(stop) if stop else ["</tool_call>"]

        self._train_fraction = float(train_fraction)
        self._buffer_max_staleness = buffer_max_staleness
        self._tail_policy = str(tail_policy)
        if self._tail_policy not in ("carry", "drop"):
            raise ValueError(f"tail_policy must be 'carry' or 'drop'; got {self._tail_policy!r}")
        self._drive_seq = 0
        self._carried_tail_trajectories = 0
        self._n = int(samples_per_prompt)
        self._per_worker_inflight = int(per_worker_inflight)
        self._worker_max_concurrency = int(cfg.get("worker_max_concurrency", 1))
        self._oversample = int(oversample_batch_size) if oversample_batch_size else self.batch_size
        if self._oversample < self.batch_size:
            raise ValueError(f"oversample_batch_size={self._oversample} must be >= batch_size={self.batch_size}")

        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train "
                f"devices of {self.num_devices}; must leave a non-empty rollout slab."
            )
        self._rollout_devices = self.num_devices - self._train_devices

        with placement(self.pool, fraction=self._train_fraction, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)
            if sync_cfg is not None:
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
        with placement(self.pool, fraction=1.0 - self._train_fraction, shared_workers=True):
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                raise ValueError(
                    "AsyncAgenticTrainer needs a dedicated agentic-rollout engine on the "
                    "separate slab (its inner sglang/vllm engine lives cross-slab)."
                )
            self.rollout = remote(**rollout_parsed)

        rules = []
        if self._tail_policy == "drop":
            rules.append(drop_incomplete)
        if self._buffer_max_staleness is not None:
            rules.append(keep_within_lag(self._buffer_max_staleness))
        self._rollout_manager = self._create_rollout_manager(chain(*rules) if rules else identity)
        self._rollout_weight_version = 0

        if self.weight_sync is not None:
            self._connect_separate(sync_cfg)

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab NCCL handshake (identical to AsyncARTrainer)."""
        target = str(sync_cfg.get("_target_", ""))
        if not target.endswith("NCCLWeightSync"):
            raise ValueError(
                f"AsyncAgenticTrainer (separate slabs) requires a cross-slab weight sync "
                f"(NCCLWeightSync); got sync._target_={target!r}."
            )
        addr, port = self.weight_sync.pick_master()[0]
        self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
        self.weight_sync.connect(
            master_addr=addr,
            master_port=port,
            num_rollout_gpus=len(self.rollout.workers),
        )

    def _build_tasks(self, carried: List[Sample], rollout_id: int) -> List[Sample]:
        """A drive's task list: ``n`` fresh siblings per new prompt + carried partials.

        Fresh prompts (``oversample_batch_size`` of them) are split into per-prompt
        request trees and replicated ``n`` times (the GRPO siblings share a root id);
        carried partials (their root ids + turn history preserved) are appended as-is;
        the harness resumes them from ``len(gen_parts())``.

        Fresh roots are namespaced by ``_drive_seq`` as well as ``rollout_id`` so a
        refill can never reuse a drained drive's root ids for different source rows.
        Carried partials keep the ids they were submitted under, so they still rejoin
        their own siblings in the assembler.
        """
        self._drive_seq += 1
        fresh = self._build_request_sample(self.data_source.get_samples(self._oversample), rollout_id)
        fresh = fresh.map_sample_ids(lambda sample_id: f"d{self._drive_seq}:{sample_id}")
        tasks = [prompt for prompt in fresh.split() for _ in range(self._n)]
        tasks.extend(carried)
        return tasks

    def _submit_drive(self, carried: List[Sample], rollout_id: int) -> None:
        self._rollout_manager.submit(self._build_tasks(carried, rollout_id))

    def _log_tail_metrics(self, rollout_step: int) -> None:
        self.wandb_logger.log_rollout(
            rollout_step,
            {
                "async/carried_tail_trajectories": self._carried_tail_trajectories,
            },
        )

    def _next_batch(self, rollout_id: int) -> List[Sample]:
        refills = 0
        while True:
            try:
                return self._rollout_manager.collect(self.batch_size)
            except RolloutUnderflow:
                refills += 1
                if refills > self._MAX_REFILLS:
                    raise RuntimeError(
                        f"async-agentic rollout {rollout_id}: buffer underflow after {refills} refills; "
                        "raise oversample_batch_size or buffer_max_staleness"
                    ) from None
                self._submit_drive([], rollout_id)

    def _train_on_trajectories(
        self, trajs: List[Sample], *, training_progress: float, rollout_id: int, t0: float
    ) -> Tuple[TrainStepResult, float]:
        """Reward + GROUP-relative advantage + one step over ``batch_size`` complete
        groups. Reuses :class:`AgenticTrainer`'s reward/GRPO/log helpers; the train-part
        assembly mirrors :meth:`AgenticTrainer.train_step` (steps 5-6)."""
        rewards, group_ids = self._rewards_and_groups(trajs, rollout_id)
        finite = torch.isfinite(rewards)
        mean_reward = float(rewards[finite].mean().item()) if bool(finite.any()) else 0.0
        advantages = self._group_advantages(rewards, group_ids)

        train_parts: List[Part] = []
        for i, tr in enumerate(trajs):
            adv_i = float(advantages[i].item())
            for gp in tr.gen_parts():
                gp = _part_with_field(gp, "advantages", torch.full((gp.batch_size,), adv_i, dtype=torch.float32))
                gp = _part_with_field(gp, "primitives", {})
                gp = _part_with_field(gp, "rewards", None)
                train_parts.append(gp)

        depths = [len(tr.gen_parts()) for tr in trajs]
        if not train_parts:
            logger.warning("AsyncAgenticTrainer rollout %d produced no trainable turns.", rollout_id)
            return TrainStepResult(0.0, 0.0, 0.0, False, [], {}), mean_reward

        train_part = self._pad_to_dp_multiple(Part.concat(train_parts))
        result = self.stack.train_track(train_part, training_progress=float(training_progress))

        log_sample = self._build_log_sample(trajs, rewards, advantages, rollout_id)
        versions = [gp.weight_version for tr in trajs for gp in tr.gen_parts() if gp.weight_version is not None]
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            log_sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics={
                "agent/mean_turns": (sum(depths) / len(depths)) if depths else 0.0,
                "agent/max_turns": max(depths) if depths else 0,
                "async/weight_version": self._rollout_weight_version,
                "async/version_span": (max(versions) - min(versions)) if versions else 0,
                "async/carried_tail_trajectories": self._carried_tail_trajectories,
            },
        )
        self._reset_transport_buffers()
        return result, mean_reward

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "full",
    ) -> None:
        interval = max(1, weight_sync_interval)

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        for _ in range(start_rollout):
            self.data_source.get_samples(self._oversample)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "adv_normalization_scope": self.adv_normalization_scope,
                "buffer_max_staleness": self._buffer_max_staleness,
                "oversample_batch_size": self._oversample,
                "train_fraction": self._train_fraction,
                "tail_policy": self._tail_policy,
                "weight_sync_interval": interval,
            },
        )

        if start_rollout < num_rollouts and start_rollout and self.weight_sync is not None:
            self._rollout_weight_version = self._rollout_manager.sync_weights(self.weight_sync)
        if start_rollout < num_rollouts:
            self._submit_drive(carried=[], rollout_id=start_rollout)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                trajs = self._next_batch(rollout_id)
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._train_on_trajectories(
                    trajs, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                need_save = save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts)
                need_sync = step % interval == 0 and self.weight_sync is not None
                if need_save or need_sync:
                    carried = self._rollout_manager.quiesce()
                    self._carried_tail_trajectories += len(carried)
                    if carried:
                        self._log_tail_metrics(step)
                    if need_save:
                        self.maybe_save_checkpoint(
                            rollout_id,
                            num_rollouts,
                            save_interval=save_interval,
                            save_dir=save_dir,
                            save_mode=save_mode,
                        )
                    if need_sync:
                        self._rollout_weight_version = self._rollout_manager.sync_weights(self.weight_sync)
                    if step < num_rollouts:
                        self._submit_drive(carried=carried, rollout_id=step)
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                carried = self._rollout_manager.quiesce()
                self._carried_tail_trajectories += len(carried)
                if carried:
                    self._log_tail_metrics(num_rollouts)
            except BaseException:  # noqa: BLE001 — preserve an active training failure
                if active_error:
                    logger.warning("AsyncAgenticTrainer cleanup failed", exc_info=True)
                else:
                    raise
            finally:
                try:
                    self._finish_wandb()
                finally:
                    self._shutdown_runtime()
