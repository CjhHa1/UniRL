"""Admission limits shared by single-loop and continuous rollout control."""

from __future__ import annotations

from typing import Optional


def next_hard_boundary(
    trained_batches: int,
    *,
    num_rollouts: int,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> int:
    boundary = num_rollouts
    for interval in (eval_interval, save_interval):
        if interval > 0 and trained_batches < num_rollouts:
            boundary = min(boundary, ((trained_batches // interval) + 1) * interval)
    return boundary


def boundary_launch_slots(
    *,
    inflight_count: int,
    ready_count: int,
    max_inflight: int,
    trained_batches: int,
    num_rollouts: int,
    hard_boundary: int,
    batches_since_sync: int,
    weight_sync_interval: int,
    max_pending: Optional[int] = None,
) -> int:
    """Generations admissible before the next publication or durable boundary."""

    freshness = weight_sync_interval - batches_since_sync
    allowed = min(freshness, min(num_rollouts, hard_boundary) - trained_batches)
    outstanding = inflight_count + ready_count
    limits = [
        max_inflight - inflight_count,
        allowed - outstanding,
    ]
    if max_pending is not None:
        limits.append(max_pending - outstanding)
    return max(0, min(limits))


__all__ = ["boundary_launch_slots", "next_hard_boundary"]
