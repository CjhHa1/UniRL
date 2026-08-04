"""Optimizer-update policy control shared by async AR and diffusion trainers.

Two quantities ride the optimizer-update clock and only one of them is staleness:

* ``staleness`` — updates between the current train weights and the behavior
  policy that generated a batch, i.e. the off-policyness of the data actually
  being trained on. This is AReaL's ``eta`` / ``max_head_offpolicyness``.
* ``publish_lag`` — updates between the current train weights and the snapshot
  last published to the rollout engine. This is sync debt: no batch is that
  stale, but it is what makes a weight sync due.

Recipes state the budget in whole rollout batches (``max_staleness``) because
admission and consumption only ever run at a batch boundary. Stating it in raw
updates instead would quantize it to ``num_updates_per_batch`` — 22 and 23 would
both mean a 12-batch depth — and silently change meaning whenever that count
changes. ``staleness_budget`` converts once into the clock the versions count in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AsyncBatchControl:
    """Track train/rollout versions and gate batch generation."""

    max_staleness: int
    num_updates_per_batch: int
    train_version: int = 0
    rollout_version: int = 0

    def __post_init__(self) -> None:
        if self.max_staleness < 0:
            raise ValueError(f"max_staleness must be >= 0, got {self.max_staleness}")
        if self.num_updates_per_batch < 1:
            raise ValueError(f"num_updates_per_batch must be >= 1, got {self.num_updates_per_batch}")
        if self.rollout_version > self.train_version:
            raise ValueError(
                f"rollout_version cannot be ahead of train_version: {self.rollout_version} > {self.train_version}"
            )

    @property
    def staleness_budget(self) -> int:
        """``max_staleness`` batches expressed in committed optimizer updates."""

        return self.max_staleness * self.num_updates_per_batch

    @property
    def publish_lag(self) -> int:
        """Sync debt: updates the published rollout snapshot trails train by."""

        return self.train_version - self.rollout_version

    @property
    def admission_depth(self) -> int:
        """Outstanding generations the budget allows, in whole batches."""

        return self.max_staleness + 1

    def restore(self, train_version: int) -> None:
        self.train_version = train_version
        self.rollout_version = 0

    def staleness(self, behavior_version: int) -> int:
        """Optimizer updates between train and a batch's behavior policy."""

        stale = self.train_version - behavior_version
        if stale < 0:
            raise ValueError(
                f"rollout batch has future behavior version {behavior_version} > train version {self.train_version}"
            )
        return stale

    def record_optimizer_updates(self, optimizer_updates: int) -> None:
        self.train_version += optimizer_updates

    def launch_slots(
        self,
        *,
        inflight_count: int,
        ready_count: int,
        max_inflight: int,
        trained_batches: int,
        num_rollouts: int,
        hard_boundary: int,
    ) -> int:
        if self.publish_lag > self.staleness_budget:
            return 0
        # Floor division is what keeps a partially-committed step honest: a step
        # that skipped updates moved the clock by less than a whole batch, and
        # rounding down never admits a generation the budget cannot cover.
        freshness = (self.staleness_budget - self.publish_lag) // self.num_updates_per_batch + 1
        allowed = min(freshness, num_rollouts - trained_batches, hard_boundary - trained_batches)
        return max(0, min(max_inflight - inflight_count, allowed - inflight_count - ready_count))

    def sync_rollout(self, engine: Any, rollout: Any, weight_sync: Any, *, force: bool = False) -> bool:
        if not force and self.publish_lag == 0:
            return False
        engine.quiesce()
        if engine.ready_count:
            raise RuntimeError(
                f"cannot sync rollout weights with completed batches queued: ready_count={engine.ready_count}"
            )
        weight_sync.sync()
        rollout.set_policy_version(self.train_version)
        self.rollout_version = self.train_version
        return True

    def behavior_metrics(self, behavior_version: int) -> dict[str, float]:
        staleness = self.staleness(behavior_version)
        return {
            "async/behavior_version": behavior_version,
            "async/staleness_updates": staleness,
            "async/staleness_batches": staleness / self.num_updates_per_batch,
        }

    def train_metrics(self, optimizer_updates: int) -> dict[str, int]:
        return {
            "async/train_version": self.train_version,
            "async/publish_lag": self.publish_lag,
            "async/optimizer_updates": optimizer_updates,
        }


def log_admission_notes(control: AsyncBatchControl, *, max_inflight: int) -> None:
    """Report admission settings whose effect differs from what the value suggests.

    All three are legitimate configurations, so none of them is an error; each is
    a case where the recipe's number does not buy what its name implies.
    """

    if control.max_staleness == 0:
        logger.warning(
            "max_staleness=0 admits one generation at a time; generation cannot overlap the preceding train batch"
        )
    if max_inflight > control.admission_depth:
        logger.warning(
            "max_inflight=%d exceeds the staleness admission depth %d; the extra concurrency cannot be used",
            max_inflight,
            control.admission_depth,
        )
    # The loop reaps before it launches, so one completed batch can sit in the
    # ready queue behind the in-flight ones; anything past that never becomes
    # concurrency, it only defers the sync (which fires after admission_depth
    # batches, when publish_lag first exceeds the budget).
    usable_depth = max_inflight + 1
    if control.admission_depth > usable_depth:
        logger.info(
            "max_staleness=%d admits %d outstanding batches but the loop holds at most %d "
            "(max_inflight=%d plus one reaped); the surplus does not deepen the pipeline, it "
            "sets the weight-sync period to %d batches",
            control.max_staleness,
            control.admission_depth,
            usable_depth,
            max_inflight,
            control.admission_depth,
        )


def next_hard_boundary(
    trained_batches: int,
    *,
    num_rollouts: int,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> int:
    """Nearest eval/checkpoint/final boundary for launch admission."""

    boundary = num_rollouts
    for interval in (eval_interval, save_interval):
        if interval > 0 and trained_batches < num_rollouts:
            boundary = min(boundary, ((trained_batches // interval) + 1) * interval)
    return boundary


def unwrap_replicated_int(value: object, *, name: str) -> int:
    """Normalize a BROADCAST return and verify all worker replicas agree."""

    if isinstance(value, (list, tuple)):
        if not value or any(not isinstance(item, int) for item in value):
            raise TypeError(f"{name} returned invalid worker values: {value!r}")
        if any(item != value[0] for item in value[1:]):
            raise RuntimeError(f"{name} disagrees across workers: {value!r}")
        return value[0]
    if not isinstance(value, int):
        raise TypeError(f"{name} returned {type(value).__name__}, expected int")
    return value


__all__ = [
    "AsyncBatchControl",
    "log_admission_notes",
    "next_hard_boundary",
    "unwrap_replicated_int",
]
