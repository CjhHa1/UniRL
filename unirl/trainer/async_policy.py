"""Shared policy-version accounting for disaggregated async trainers.

Versions in this module count committed optimizer updates, never weight-sync
calls or consumed rollout batches.  A rollout batch records the train version
whose weights were resident in the rollout engine when generation started.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyVersionState:
    """Driver-owned train/rollout policy clocks.

    ``train_version`` advances by the number of optimizer steps that actually
    committed. ``rollout_version`` is assigned to the exact train snapshot most
    recently synced to the rollout engine.
    """

    train_version: int = 0
    rollout_version: int = 0

    def __post_init__(self) -> None:
        self.train_version = int(self.train_version)
        self.rollout_version = int(self.rollout_version)
        if self.train_version < 0 or self.rollout_version < 0:
            raise ValueError("policy versions must be non-negative")
        if self.rollout_version > self.train_version:
            raise ValueError(
                f"rollout_version cannot be ahead of train_version: {self.rollout_version} > {self.train_version}"
            )

    @property
    def rollout_lag(self) -> int:
        return self.train_version - self.rollout_version

    def record_optimizer_updates(self, committed_updates: int) -> int:
        """Advance by optimizer steps that successfully changed train weights."""

        committed_updates = int(committed_updates)
        if committed_updates < 0:
            raise ValueError(f"committed optimizer updates must be >= 0, got {committed_updates}")
        self.train_version += committed_updates
        return self.train_version

    def mark_rollout_synced(self, train_version: int) -> int:
        """Record the train version loaded by a successful rollout weight sync."""

        train_version = int(train_version)
        if train_version < self.rollout_version:
            raise ValueError(f"synced train version cannot move backwards: {train_version} < {self.rollout_version}")
        if train_version > self.train_version:
            raise ValueError(f"cannot sync future train version {train_version} > {self.train_version}")
        self.rollout_version = train_version
        return self.rollout_version

    def behavior_lag(self, behavior_version: int) -> int:
        """Optimizer-update lag between train and a batch's behavior policy."""

        behavior_version = int(behavior_version)
        lag = self.train_version - behavior_version
        if lag < 0:
            raise ValueError(
                f"rollout batch has future behavior version {behavior_version} > train version {self.train_version}"
            )
        return lag


def launch_slots(
    *,
    train_version: int,
    rollout_version: int,
    num_updates_per_batch: int,
    max_policy_lag: int,
    inflight_count: int,
    ready_count: int,
    max_inflight: int,
    trained_batches: int,
    num_rollouts: int,
    hard_boundary: int,
) -> int:
    """How many batch generations may be launched by the single-threaded loop.

    Freshness is measured in committed optimizer updates. The first outstanding
    batch would train at the current lag; every additional outstanding batch is
    conservatively reserved ``num_updates_per_batch`` future updates. Capacity and
    durable-boundary clamps are applied in the same generation-batch unit.
    """

    train = _non_negative("train_version", train_version)
    rollout = _non_negative("rollout_version", rollout_version)
    updates = _positive("num_updates_per_batch", num_updates_per_batch)
    max_lag = _non_negative("max_policy_lag", max_policy_lag)
    active = _non_negative("inflight_count", inflight_count)
    queued = _non_negative("ready_count", ready_count)
    max_active = _positive("max_inflight", max_inflight)
    trained = _non_negative("trained_batches", trained_batches)
    total = _non_negative("num_rollouts", num_rollouts)
    boundary = _non_negative("hard_boundary", hard_boundary)
    if rollout > train:
        raise ValueError(f"rollout_version cannot be ahead of train_version: {rollout} > {train}")
    if active > max_active:
        raise ValueError(f"inflight_count={active} exceeds max_inflight={max_active}")
    if boundary < trained:
        raise ValueError(f"hard_boundary={boundary} is behind trained_batches={trained}")
    if trained >= total:
        return 0

    current_lag = train - rollout
    if current_lag > max_lag:
        return 0

    freshness_slots = (max_lag - current_lag) // updates + 1
    allowed_outstanding = min(
        freshness_slots,
        total - trained,
        boundary - trained,
    )
    outstanding = active + queued
    return max(
        0,
        min(
            max_active - active,
            allowed_outstanding - outstanding,
        ),
    )


def next_hard_boundary(
    trained_batches: int,
    *,
    num_rollouts: int,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> int:
    """Nearest eval/checkpoint/final boundary for launch admission."""

    trained = _non_negative("trained_batches", trained_batches)
    total = _non_negative("num_rollouts", num_rollouts)
    boundary = total
    for interval in (int(eval_interval), int(save_interval)):
        if interval > 0 and trained < total:
            boundary = min(boundary, ((trained // interval) + 1) * interval)
    return boundary


def unwrap_replicated_int(value: object, *, name: str) -> int:
    """Normalize a BROADCAST return and verify all worker replicas agree."""

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} returned no worker values")
        first = int(value[0])
        if any(int(item) != first for item in value[1:]):
            raise RuntimeError(f"{name} disagrees across workers: {value!r}")
        return first
    return int(value)


def _non_negative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


__all__ = [
    "PolicyVersionState",
    "launch_slots",
    "next_hard_boundary",
    "unwrap_replicated_int",
]
