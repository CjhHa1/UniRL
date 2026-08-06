"""Colocate partial-rollout agentic trainers (LIN-531).

The **synchronous / colocate** driver of partial rollout — the missing quadrant next to the
barrier `AgenticTrainer` (colocate, waits for every trajectory) and the fully-async
`AsyncAgenticTrainer` (disaggregated slabs). Train and rollout still **time-share each GPU**
(`sleep()`/`wake_up()`), single loop, one optimizer step per rollout; the only change from the
barrier is *how the rollout completes*:

- **Over-sample** `oversample_batch_size` prompt-groups per drive, **commit the freshest
  `batch_size` complete GRPO groups**, and **`abort` the in-flight tail at a turn boundary**
  instead of draining every trajectory. The slow straggler no longer gates the round.
- **Tail policy** — `carry` re-submits the checkpointed tail next round when the environment can
  reconstruct its state from the `Sample` (current stateless tools); `drop` discards it for
  **stateful** envs like ALFWorld, whose `reset` starts a fresh episode.

Motivation (LIN-531 ALFWorld comparison): the fully-async trainer *lost* on ALFWorld because
disaggregation halves the generation GPUs; colocate+partial keeps all GPUs for generation while
still cutting the straggler tail — the slime "over-sample + abort + recycle" / verl `bypass_mode`
pattern. The driver-side :class:`~unirl.rollout.engine.asynchronous.AsyncAgenticRolloutEngine`
(submit/poll/finalize_if_drained/quiesce + group assembly and versioned buffering) is shared with
``AsyncAgenticTrainer``; only the colocate wake/sync/sleep choreography is new.

Correctness: `TensorWeightSync.sync` writes the live SRT weight pool, so it must run **awake +
decode-idle** — sync sits at the top (post-`wake_up`, pre-`submit`, barrier parity) and `abort`
(not `sleep`) provides the pre-sleep quiesce. The off-policy carried tail is corrected per-token
(each gen Part keeps its own `weight_version` + logprobs); `buffer_max_staleness` separately bounds
how long a completed group remains eligible in the buffer.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from typing import List, Literal, Optional

from unirl.rollout.manager import RolloutUnderflow, chain, drop_incomplete, identity, keep_within_lag
from unirl.trainer.agentic import AgenticTrainer
from unirl.trainer.agentic_env import _EnvRewardSource
from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class AgenticPartialTrainer(AgenticTrainer):
    """Colocate partial-rollout trainer (over-sample → commit-N → abort tail → carry/drop)."""

    _MAX_REFILLS = 64  # underflow guard: refills of a drained-but-short buffer before giving up

    def __init__(
        self,
        *,
        samples_per_prompt: int,
        oversample_batch_size: Optional[int] = None,
        buffer_max_staleness: Optional[int] = None,
        tail_policy: Literal["carry", "drop"] = "carry",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._n = int(samples_per_prompt)
        self._oversample = int(oversample_batch_size) if oversample_batch_size else int(self.batch_size)
        if self._oversample < self.batch_size:
            raise ValueError(f"oversample_batch_size={self._oversample} must be >= batch_size={self.batch_size}")
        self._buffer_max_staleness = buffer_max_staleness
        self._tail_policy = str(tail_policy)
        if self._tail_policy not in ("carry", "drop"):
            raise ValueError(f"tail_policy must be 'carry' or 'drop'; got {self._tail_policy!r}")
        self._drive_seq = 0
        self._carried: List[Sample] = []
        rules = []
        if self._tail_policy == "drop":
            rules.append(drop_incomplete)
        if self._buffer_max_staleness is not None:
            rules.append(keep_within_lag(self._buffer_max_staleness))
        self._rollout_manager = self._create_rollout_manager(chain(*rules) if rules else identity)
        self._rollout_weight_version = 0

    def _build_tasks(self, carried: List[Sample], rollout_id: int) -> List[Sample]:
        """A drive's task list: uniquely namespaced fresh siblings + carried partials."""
        self._drive_seq += 1
        fresh = self._build_request_sample(self.data_source.get_samples(self._oversample), rollout_id)
        fresh = fresh.map_sample_ids(lambda sample_id: f"d{self._drive_seq}:{sample_id}")
        tasks = [prompt for prompt in fresh.split() for _ in range(self._n)]
        tasks.extend(carried)
        return tasks

    def _collect_until(self, batch_size: int, rollout_id: int) -> List[List[Sample]]:
        refills = 0
        while True:
            try:
                return self._rollout_manager.collect(batch_size)
            except RolloutUnderflow:
                refills += 1
                if refills > self._MAX_REFILLS:
                    raise RuntimeError(
                        f"colocate-partial rollout {rollout_id}: buffer underflow after {refills} refills; "
                        "raise oversample_batch_size or buffer_max_staleness"
                    ) from None
                self._rollout_manager.submit(self._build_tasks([], rollout_id))

    def _drive_partial(self, rollout_id: int, sync_weights: bool) -> List[List[Sample]]:
        self.rollout.wake_up()
        active_error: Optional[BaseException] = None
        try:
            if sync_weights and self.weight_sync is not None:
                self._rollout_weight_version = self._rollout_manager.sync_weights(self.weight_sync)
            tasks = self._build_tasks(self._carried, rollout_id)
            self._carried = []
            self._rollout_manager.submit(tasks)
            groups = self._collect_until(self.batch_size, rollout_id)
            self._carried = self._rollout_manager.quiesce()
        except BaseException as exc:
            active_error = exc
            try:
                self._carried = self._rollout_manager.quiesce()
            except BaseException:
                logger.exception("AgenticPartialTrainer rollout cleanup failed")
            raise
        finally:
            try:
                self.rollout.sleep()
            except BaseException:
                if active_error is None:
                    raise
                logger.exception("AgenticPartialTrainer rollout sleep failed")
        carried = self._carried
        tail_depths = [len(t.gen_parts()) for t in carried]
        logger.info(
            "rollout %d partial: committed %d groups; %s tail=%d trajectories, turns=%s",
            rollout_id,
            len(groups),
            self._tail_policy,
            len(carried),
            dict(sorted(Counter(tail_depths).items())),
        )
        return groups

    # Train loop — override ARTrainer.train (the tail must carry across rollouts)

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        for _ in range(start_rollout):
            self.data_source.get_samples(self._oversample)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "adv_normalization_scope": self.adv_normalization_scope,
                "oversample_batch_size": self._oversample,
                "buffer_max_staleness": self._buffer_max_staleness,
                "tail_policy": self._tail_policy,
                "weight_sync_interval": interval,
            },
        )

        self._carried = []

        try:
            if self.eval_interval > 0:
                self.evaluate(rollout_id=-1)
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                training_progress = rollout_id / max(1, num_rollouts - 1)
                sync_weights = (rollout_id > 0 and rollout_id % interval == 0) or (
                    resumed and rollout_id == start_rollout
                )

                groups = self._drive_partial(rollout_id, sync_weights)
                trajs: List[Sample] = [t for group in groups for t in group]
                rewards, group_ids = self._rewards_and_groups(trajs, rollout_id)
                result, mean_reward = self._advantage_train_and_log(
                    trajs,
                    rewards,
                    group_ids,
                    rollout_id=rollout_id,
                    training_progress=training_progress,
                    t0=t0,
                    extra_metrics={
                        "partial/committed_groups": len(groups),
                        "partial/carried_trajectories": len(self._carried),
                        "partial/weight_version": self._rollout_weight_version,
                    },
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id=rollout_id)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                self._carried = self._rollout_manager.quiesce()
            except BaseException:  # noqa: BLE001 — preserve an active training failure
                if active_error:
                    logger.warning("AgenticPartialTrainer cleanup failed", exc_info=True)
                else:
                    raise
            finally:
                try:
                    self._finish_wandb()
                finally:
                    self._shutdown_runtime()


class AgenticEnvPartialTrainer(_EnvRewardSource, AgenticPartialTrainer):
    """Colocate partial-rollout trainer whose reward is the environment's per-trajectory return
    (ALFWorld etc.). Reward SOURCE only (the shared
    :class:`~unirl.trainer.agentic_env._EnvRewardSource`) — the partial-rollout machinery is
    inherited. Recipes use ``tail_policy: drop`` because stateful-episode envs restart a carried
    partial."""


__all__ = ["AgenticPartialTrainer", "AgenticEnvPartialTrainer"]
