"""Optimizer-update policy control shared by async AR and diffusion trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AsyncBatchControl:
    """Track train/rollout versions and gate batch generation."""

    max_policy_lag: int
    num_updates_per_batch: int
    train_version: int = 0
    rollout_version: int = 0

    def __post_init__(self) -> None:
        if self.max_policy_lag < 0:
            raise ValueError(f"max_policy_lag must be >= 0, got {self.max_policy_lag}")
        if self.num_updates_per_batch < 1:
            raise ValueError(f"num_updates_per_batch must be >= 1, got {self.num_updates_per_batch}")
        if self.rollout_version > self.train_version:
            raise ValueError(
                f"rollout_version cannot be ahead of train_version: {self.rollout_version} > {self.train_version}"
            )

    @property
    def rollout_lag(self) -> int:
        return self.train_version - self.rollout_version

    @property
    def freshness_depth(self) -> int:
        return self.max_policy_lag // self.num_updates_per_batch + 1

    def restore(self, train_version: int) -> None:
        self.train_version = train_version
        self.rollout_version = 0

    def behavior_lag(self, behavior_version: int) -> int:
        lag = self.train_version - behavior_version
        if lag < 0:
            raise ValueError(
                f"rollout batch has future behavior version {behavior_version} > train version {self.train_version}"
            )
        return lag

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
        if self.rollout_lag > self.max_policy_lag:
            return 0
        freshness = (self.max_policy_lag - self.rollout_lag) // self.num_updates_per_batch + 1
        allowed = min(freshness, num_rollouts - trained_batches, hard_boundary - trained_batches)
        return max(0, min(max_inflight - inflight_count, allowed - inflight_count - ready_count))

    def sync_rollout(self, engine: Any, rollout: Any, weight_sync: Any, *, force: bool = False) -> bool:
        if not force and self.rollout_lag == 0:
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

    def behavior_metrics(self, behavior_version: int) -> dict[str, int]:
        return {
            "async/behavior_version": behavior_version,
            "async/behavior_lag": self.behavior_lag(behavior_version),
        }

    def train_metrics(self, optimizer_updates: int) -> dict[str, int]:
        return {
            "async/train_version": self.train_version,
            "async/rollout_lag": self.rollout_lag,
            "async/optimizer_updates": optimizer_updates,
        }


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


__all__ = ["AsyncBatchControl", "next_hard_boundary", "unwrap_replicated_int"]
