"""Async autoregressive RL trainer — disaggregated train/rollout slabs.

Sibling of :class:`~unirl.trainer.ar.ARTrainer` (synchronous + *colocated*:
rollout engine and FSDP train shard time-share each GPU via ``sleep()/wake_up()``,
and every step runs ``generate → reward → train`` in series). ``AsyncARTrainer``
instead places training and rollout on **disjoint GPU slabs**, keeps the engine
**resident**, pushes weights cross-slab via ``NCCLWeightSync``, and overlaps
generation with training.

ONE single-threaded loop (slime's "one trainer loop; async-depth is a knob"
principle, implemented with UniRL-native non-blocking Ray dispatch instead of
slime's thread+asyncio). Async freshness is measured directly in committed
optimizer updates:

* ``max_inflight`` — how many generations run concurrently (overlap/parallelism
  depth). ``1`` ≈ the classic one-step pipeline; higher fans out more.
* ``max_policy_lag`` — inclusive train-minus-behavior optimizer-update
  lag at batch admission. ``0`` aligns policy versions; the rollout-anchored
  PPO ratio remains the numerical source of truth.

Generation runs through :class:`~unirl.rollout.engine.asynchronous.AsyncBatchRolloutEngine`
(non-blocking Ray futures over the rollout Handle) on the single driver thread —
no producer thread, no locks; the trainer's ``_next_rollout_batch`` loop owns the policy
(optimizer-update launch admission, launch-then-reap order). Draining all in-flight generations
before each weight sync is **mandatory** (the engine corrupts an in-flight
generation when weights + KV cache update mid-flight); this is the
single-threaded ``_drain_all`` quiesce.

Subclasses ``ARTrainer`` to reuse ``_build_request_sample``/``evaluate`` and ``BaseTrainer``
plumbing, but ``__init__`` calls ``BaseTrainer.__init__`` **directly** (the parent
opens the colocate ``placement(fraction=1.0)`` block we replace with two slabs).
"""

import inspect
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.models.qwen3_5.validation import validate_qwen3_5_training_contract
from unirl.rollout.engine.asynchronous import AsyncBatchRolloutEngine, RolloutBatch
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer
from unirl.trainer.async_policy import PolicyVersionState, launch_slots, next_hard_boundary, unwrap_replicated_int
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


def _rollout_dp_size_from_parsed_config(rollout_parsed: dict, *, world_size: int) -> int:
    """Compute the rollout Handle's DP width before constructing GPU roles."""
    from unirl.distributed.group.handle import _parallel_shape_from_init_kwargs

    role_cls = rollout_parsed["role_cls"]
    init_kwargs = {key: value for key, value in rollout_parsed.items() if key != "role_cls"}
    sp_size, tp_size, pp_size, _ = _parallel_shape_from_init_kwargs(
        init_kwargs,
        int(world_size),
        role_cls,
    )
    non_dp_width = sp_size if sp_size > 1 else tp_size * pp_size
    return int(world_size) // int(non_dp_width)


class AsyncARTrainer(ARTrainer):
    """Disaggregated async AR trainer (two slabs, resident engine, NCCL sync)."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
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
        advantage_mode: str = "grpo",
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = -1,
        eval_batch_size: int = 8,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        train_fraction: float = 0.5,
        max_inflight: int = 1,
        max_policy_lag: int = 0,
    ) -> None:
        validate_qwen3_5_training_contract(
            pipeline_cfg=pipeline_cfg,
            backend_cfg=backend_cfg,
            rollout_cfg=rollout_cfg,
            stack_cfg=stack_cfg,
        )
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        self.batch_size = batch_size
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.advantage_mode = str(advantage_mode).strip().lower()
        if self.advantage_mode not in ("grpo", "gae"):
            raise ValueError(f"AsyncARTrainer: advantage_mode must be 'grpo' or 'gae', got {advantage_mode!r}")
        self.balance_shards = bool(balance_shards)
        self.eval_interval = int(eval_interval)
        _num = int(eval_num_prompts)
        self.eval_num_prompts = -1 if _num < 0 else _num
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self.weight_sync = None
        rollout_parsed = parse_hydra_cfg(rollout_cfg)
        if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
            raise ValueError(
                "AsyncARTrainer needs a dedicated-rollout engine (vllm/sglang) on the "
                "separate slab; the trainside direct-sampling engine needs the pipeline "
                "as a local sibling and cannot live cross-slab."
            )
        self._rollout_anchor_device = None

        self._train_fraction = float(train_fraction)
        self._max_inflight = max(1, int(max_inflight))
        self._max_policy_lag = max_policy_lag
        self._num_updates_per_batch = stack_cfg.get("num_updates_per_batch", 1)
        if self._max_policy_lag < 0:
            raise ValueError(f"max_policy_lag must be >= 0, got {self._max_policy_lag}")
        if self._num_updates_per_batch < 1:
            raise ValueError(f"stack.num_updates_per_batch must be >= 1, got {self._num_updates_per_batch}")
        freshness_depth = self._max_policy_lag // self._num_updates_per_batch + 1
        if self._max_inflight > freshness_depth:
            logger.warning(
                "max_inflight=%d exceeds the policy-lag admission depth %d; the extra concurrency cannot be used",
                self._max_inflight,
                freshness_depth,
            )
        if freshness_depth == 1:
            logger.warning(
                "async policy-lag settings admit one generation at a time; "
                "generation cannot overlap the preceding train batch"
            )
        self._policy_versions = PolicyVersionState()
        self._rollout_initialized = False
        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train "
                f"devices of {self.num_devices}; must leave a non-empty rollout slab."
            )
        # Require rollout batches to divide evenly across train and rollout slabs.
        self._rollout_devices = self.num_devices - self._train_devices
        prompts = int(self.batch_size)
        total = prompts * total_samples_per_prompt(self.sampling_params)
        if total % self._train_devices != 0:
            raise ValueError(
                f"batch_size * samples_per_prompt = {total} is not divisible by the train "
                f"slab size {self._train_devices}; adjust batch_size / samples_per_prompt / train_fraction."
            )
        rollout_dp_size = _rollout_dp_size_from_parsed_config(
            rollout_parsed,
            world_size=self._rollout_devices,
        )
        if prompts % rollout_dp_size != 0:
            raise ValueError(
                f"batch_size = {prompts} prompts is not divisible by the rollout DP size "
                f"{rollout_dp_size} ({self._rollout_devices} rollout GPUs; each prompt-tree "
                "DP-scatters whole); adjust batch_size / train_fraction / rollout TP."
            )

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
            self.rollout = remote(**rollout_parsed)

        if self.weight_sync is None:
            raise ValueError("AsyncARTrainer requires a cross-slab weight sync; add a `sync:` block.")
        self._connect_separate(sync_cfg)

    def _prepare_rollout(self, *, sync_weights: bool) -> bool:
        """Sync a resident separate-slab engine without colocate handoffs."""
        if sync_weights:
            self._sync_rollout_weights()
        return False

    def _finish_rollout(self, *, train_state_offloaded: bool) -> None:
        """Keep the separate rollout slab resident across train/eval phases."""
        if train_state_offloaded:
            raise RuntimeError("AsyncARTrainer cannot offload its disjoint training slab during rollout")

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab handshake (NCCL branch of diffusion.py:191-208).

        Rank 0 picks a rendezvous addr/port, is handed the rollout slab's Worker
        actor handles, then ``connect`` fires each rollout worker's
        ``init_weights_update_group`` non-blocking and joins the broadcast group
        itself. Only ``NCCLWeightSync`` is supported here (always cross-slab
        full-weight); a non-NCCL target is a config error.
        """
        target = str(sync_cfg.get("_target_", ""))
        if not target.endswith("NCCLWeightSync"):
            raise ValueError(
                f"AsyncARTrainer (separate slabs) requires a cross-slab weight sync "
                f"(NCCLWeightSync); got sync._target_={target!r}."
            )
        addr, port = self.weight_sync.pick_master()[0]
        tp_size = self.rollout.tp_size
        pp_size = self.rollout.pp_size
        targets = self.rollout.tp_zero_workers
        self.weight_sync.set_rollout_targets(targets, self.rollout.role_name)
        self.weight_sync.connect(
            master_addr=addr,
            master_port=port,
            num_rollout_gpus=len(targets) * tp_size,
            tp_size=tp_size,
            pp_size=pp_size,
        )

    def _build_async_sample(self, gen_id: int) -> Sample:
        """Consume one data batch and build the request Sample for ``gen_id``."""
        return self._build_request_sample(self.data_source.get_samples(self.batch_size), gen_id)

    def _score_completed(self, gen_id: int, completed: Sample) -> List[Sample]:
        """Score a completed Sample and split it into tree-complete groups.

        Runs at reap time inside the engine. Scoring must precede
        ``_drop_decoded`` (the reward reads the decoded primitive). Keyed by
        ``gen_id`` so media panels behave like the old path. The filled
        ``Sample`` is self-contained (it carries its input Parts).
        """
        scored = self.reward.score_and_attach(completed)
        self._drop_decoded(scored, rollout_id=gen_id)
        return scored.split()

    def _drain_all(self) -> None:
        """Finish + buffer EVERY in-flight generation (the single-threaded quiesce).

        Mandatory before a weight sync (the engine corrupts an in-flight generate
        when weights + KV cache update mid-flight), before eval/checkpoint (shared
        engine), and in ``finally`` (no leaked ObjectRefs).
        """
        self._async_engine.quiesce()

    def _sync_rollout_weights(self, *, force: bool = False) -> bool:
        """Load the current train weights into an empty rollout engine."""

        versions = self._policy_versions
        if not force and self._rollout_initialized and versions.rollout_version == versions.train_version:
            return False
        self._drain_all()
        if self._async_engine.ready_count != 0:
            raise RuntimeError(
                f"cannot sync rollout weights with completed batches queued: "
                f"ready_count={self._async_engine.ready_count}"
            )
        target = versions.train_version
        self.weight_sync.sync()
        self.rollout.set_policy_version(target)
        versions.mark_rollout_synced(target)
        self._rollout_initialized = True
        return True

    def _policy_metrics(self, batch: RolloutBatch) -> Dict[str, float]:
        versions = self._policy_versions
        return {
            "async/behavior_version": batch.behavior_version,
            "async/behavior_lag": versions.behavior_lag(batch.behavior_version),
        }

    def _advantage_and_train(
        self,
        sample: Sample,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[TrainStepResult, float]:
        """Advantage + optimizer updates for a scored ``Sample`` (rewards already attached)."""
        if t0 is None:
            t0 = time.perf_counter()
        part = sample.parts[-1]
        mean_reward = 0.0
        if part.rewards is not None:
            part.rewards = hydrate(part.rewards)
            mean_reward = float(part.rewards.to(torch.float32).mean().item())
        if self.advantage_mode == "grpo":
            part = part.compute_advantages(
                normalize=self.normalize_adv_by_std,
                scope=self.adv_normalization_scope,
            )
        sample = sample.with_parts([*sample.parts[:-1], part])
        train_part = part
        if self.balance_shards:
            train_part = part.balance_shards(self._train_devices)
        result = self.stack.train_track(train_part, training_progress=float(training_progress))
        self._policy_versions.record_optimizer_updates(result.optimizer_updates)
        if extra_metrics is not None:
            extra_metrics.update(
                {
                    "async/train_version": self._policy_versions.train_version,
                    "async/rollout_lag": self._policy_versions.rollout_lag,
                    "async/optimizer_updates": result.optimizer_updates,
                }
            )
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            sample,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params.get("ar"), "max_new_tokens", None),
            extra_metrics=extra_metrics,
        )
        self._reset_transport_buffers()
        return result, mean_reward

    def train(
        self,
        *,
        num_rollouts: int,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "full",
    ) -> None:
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        train_version = unwrap_replicated_int(
            self.backend.get_optimizer_step_count(),
            name="backend optimizer step count",
        )
        self._policy_versions = PolicyVersionState(train_version=train_version)
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "adv_normalization_scope": self.adv_normalization_scope,
                "max_inflight": self._max_inflight,
                "max_policy_lag": self._max_policy_lag,
                "num_updates_per_batch": self._num_updates_per_batch,
            },
        )

        self._async_engine = AsyncBatchRolloutEngine(
            self.rollout,
            process_completion=self._score_completed,
            groups_per_batch=self.batch_size,
            start_gen_id=start_rollout,
        )

        if resumed:
            self._sync_rollout_weights(force=True)
        if self.eval_interval > 0:
            self.evaluate(rollout_id=-1)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                hard_boundary = next_hard_boundary(
                    rollout_id,
                    num_rollouts=num_rollouts,
                    eval_interval=self.eval_interval,
                    save_interval=save_interval,
                )
                batch = self._next_rollout_batch(
                    rollout_id,
                    num_rollouts=num_rollouts,
                    hard_boundary=hard_boundary,
                )
                sample = Sample.concat(batch.groups)
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    sample,
                    training_progress=training_progress,
                    rollout_id=rollout_id,
                    t0=t0,
                    extra_metrics=self._policy_metrics(batch),
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                eval_due = self.eval_interval > 0 and step % self.eval_interval == 0
                save_due = save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts)
                sync_due = step < num_rollouts and self._policy_versions.rollout_lag > self._max_policy_lag
                if eval_due or save_due or sync_due:
                    if self._async_engine.inflight_count + self._async_engine.ready_count != 0:
                        raise RuntimeError(
                            "async sync boundary retained rollout work: "
                            f"inflight_count={self._async_engine.inflight_count}, "
                            f"ready_count={self._async_engine.ready_count}"
                        )
                    self._sync_rollout_weights()

                if eval_due:
                    self.evaluate(rollout_id=rollout_id)
                if save_due:
                    self.maybe_save_checkpoint(
                        rollout_id,
                        num_rollouts,
                        save_interval=save_interval,
                        save_dir=save_dir,
                        save_mode=save_mode,
                    )
        finally:
            active_exception = sys.exc_info()[0] is not None
            try:
                self._drain_all()
            except Exception:
                if not active_exception:
                    raise
                logger.exception("Failed to drain in-flight generations during trainer teardown")
            finally:
                self._finish_wandb()

    def _next_rollout_batch(
        self,
        rollout_id: int,
        *,
        num_rollouts: int,
        hard_boundary: int,
    ) -> RolloutBatch:
        """Launch, reap, and consume one completion-order FIFO train batch."""

        engine = self._async_engine
        while True:
            slots = launch_slots(
                train_version=self._policy_versions.train_version,
                rollout_version=self._policy_versions.rollout_version,
                num_updates_per_batch=self._num_updates_per_batch,
                max_policy_lag=self._max_policy_lag,
                inflight_count=engine.inflight_count,
                ready_count=engine.ready_count,
                max_inflight=self._max_inflight,
                trained_batches=rollout_id,
                num_rollouts=num_rollouts,
                hard_boundary=hard_boundary,
            )
            for _ in range(slots):
                engine.submit(
                    self._build_async_sample(engine.next_gen_id),
                    behavior_version=self._policy_versions.rollout_version,
                )
            engine.poll()
            batch = engine.pop_next_batch(
                train_version=self._policy_versions.train_version,
                max_policy_lag=self._max_policy_lag,
            )
            if batch is not None:
                return batch
            if engine.inflight_count:
                engine.wait_oldest()
            else:
                raise RuntimeError("async rollout queue is empty and policy lag admits no new generation")
