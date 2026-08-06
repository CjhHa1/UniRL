"""Optimizer-update policy control shared by async AR and diffusion trainers.

Two quantities ride the optimizer-update clock and only one of them is staleness:

* ``staleness`` — updates between the output version that generated a batch and
  the train weights that batch starts training against, i.e. the off-policyness
  of the data.
* ``publish_lag`` — updates between the current train weights and the snapshot
  last published to the rollout engine. This records sync debt separately from
  the batch-counted publication cadence.

Recipes state the publication cadence as ``weight_sync_interval`` rollout
batches. One published rollout snapshot serves that many batches, and the
oldest batch admitted under it is ``weight_sync_interval - 1`` batches stale.
``staleness_budget`` converts that derived maximum into the committed-optimizer-
update clock used by the version ledger.

Batch entry is also the only point the budget is enforced at, which matters once
``num_updates_per_batch > 1``: the anchor is frozen for the whole batch while the
weights keep moving, so update ``i`` trains at ``staleness + i - 1`` and the worst
case any gradient step sees is ``staleness_budget + num_updates_per_batch - 1``.
That extra span is the in-batch off-policyness PPO already assumes — the frozen
anchor and ``clip_range`` cover it — so it is deliberately outside the budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AsyncBatchControl:
    """Track train/published versions and gate batch generation."""

    weight_sync_interval: int
    num_updates_per_batch: int
    train_version: int = 0
    published_version: int = 0
    batches_since_sync: int = 0

    def __post_init__(self) -> None:
        if self.weight_sync_interval < 1:
            raise ValueError(f"weight_sync_interval must be >= 1, got {self.weight_sync_interval}")
        if self.num_updates_per_batch < 1:
            raise ValueError(f"num_updates_per_batch must be >= 1, got {self.num_updates_per_batch}")
        if self.published_version > self.train_version:
            raise ValueError(
                f"published_version cannot be ahead of train_version: {self.published_version} > {self.train_version}"
            )
        if not 0 <= self.batches_since_sync <= self.weight_sync_interval:
            raise ValueError(
                "batches_since_sync must be within the current publication interval: "
                f"0 <= {self.batches_since_sync} <= {self.weight_sync_interval}"
            )

    @property
    def max_staleness(self) -> int:
        """Maximum batch-entry staleness implied by the publication cadence."""

        return self.weight_sync_interval - 1

    @property
    def staleness_budget(self) -> int:
        """Derived maximum staleness expressed in committed optimizer updates."""

        return self.max_staleness * self.num_updates_per_batch

    @property
    def publish_lag(self) -> int:
        """Sync debt: updates the published rollout snapshot trails train by."""

        return self.train_version - self.published_version

    @property
    def admission_depth(self) -> int:
        """Batches one published rollout snapshot may serve."""

        return self.weight_sync_interval

    def restore(self, train_version: int) -> None:
        self.train_version = train_version
        self.published_version = 0
        self.batches_since_sync = 0

    def staleness(self, output_version: int) -> int:
        """Optimizer updates between train and the policy that produced a batch."""

        stale = self.train_version - output_version
        if stale < 0:
            raise ValueError(
                f"rollout batch has future output version {output_version} > train version {self.train_version}"
            )
        return stale

    def record_optimizer_updates(self, optimizer_updates: int) -> None:
        self.train_version += optimizer_updates
        self.batches_since_sync += 1

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
        if self.batches_since_sync >= self.weight_sync_interval:
            return 0
        freshness = self.weight_sync_interval - self.batches_since_sync
        allowed = min(freshness, num_rollouts - trained_batches, hard_boundary - trained_batches)
        return max(0, min(max_inflight - inflight_count, allowed - inflight_count - ready_count))

    def sync_rollout(self, engine: Any, rollout: Any, weight_sync: Any, *, force: bool = False) -> bool:
        if not force and self.publish_lag == 0:
            self.batches_since_sync = 0
            return False
        engine.quiesce()
        if engine.ready_count:
            raise RuntimeError(
                f"cannot sync rollout weights with completed batches queued: ready_count={engine.ready_count}"
            )
        weight_sync.sync()
        rollout.set_version(self.train_version)
        self.published_version = self.train_version
        self.batches_since_sync = 0
        return True

    def output_metrics(self, output_version: int) -> dict[str, float]:
        staleness = self.staleness(output_version)
        return {
            "async/output_version": output_version,
            "async/staleness_updates": staleness,
            "async/staleness_batches": staleness / self.num_updates_per_batch,
        }

    def train_metrics(self, optimizer_updates: int) -> dict[str, int]:
        return {
            "async/train_version": self.train_version,
            "async/publish_lag": self.publish_lag,
            "async/optimizer_updates": optimizer_updates,
            "async/batches_since_sync": self.batches_since_sync,
        }


def sync_period_batches(
    control: AsyncBatchControl,
    *,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> int:
    """Batches between weight publications, once every admission limit is applied.

    The configured interval would publish every ``admission_depth`` batches,
    but :func:`next_hard_boundary` clamps admission as well, so an eval or
    checkpoint interval below that depth becomes the period instead.
    """

    period = control.admission_depth
    for interval in (eval_interval, save_interval):
        if interval > 0:
            period = min(period, interval)
    return period


def log_admission_notes(
    control: AsyncBatchControl,
    *,
    max_inflight: int,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> None:
    """Report admission settings whose effect differs from what the value suggests.

    All of these are legitimate configurations, so none is an error; each is a
    case where the recipe's number does not buy what its name implies.
    """

    period = sync_period_batches(control, eval_interval=eval_interval, save_interval=save_interval)

    if control.weight_sync_interval == 1:
        logger.warning(
            "weight_sync_interval=1 admits one generation at a time; generation cannot overlap the preceding "
            "train batch"
        )
    if max_inflight > control.admission_depth:
        logger.warning(
            "max_inflight=%d exceeds the staleness admission depth %d; the extra concurrency cannot be used",
            max_inflight,
            control.admission_depth,
        )
    # The loop reaps before it launches, so one completed batch can sit in the
    # ready queue behind the in-flight ones; anything past that never becomes
    # concurrency, it only defers the sync.
    usable_depth = max_inflight + 1
    if control.admission_depth > usable_depth:
        logger.info(
            "weight_sync_interval=%d admits %d outstanding batches but the loop holds at most %d "
            "(max_inflight=%d plus one reaped); the surplus does not deepen the pipeline, it "
            "sets the weight-sync period to %d batches",
            control.weight_sync_interval,
            control.admission_depth,
            usable_depth,
            max_inflight,
            period,
        )
    if period < control.admission_depth:
        logger.warning(
            "eval/checkpoint boundaries publish every %d batches, below weight_sync_interval=%d "
            "(eval_interval=%d, save_interval=%d), so derived max_staleness=%d is never fully "
            "spent — data tops out at %d batches stale",
            period,
            control.admission_depth,
            eval_interval,
            save_interval,
            control.max_staleness,
            period - 1,
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
    "sync_period_batches",
    "unwrap_replicated_int",
]
