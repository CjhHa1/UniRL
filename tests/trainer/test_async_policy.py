from __future__ import annotations

import pytest

from unirl.train.stack.base import TrainStepResult, _aggregate_update_results
from unirl.trainer.async_policy import PolicyVersionState, launch_slots, next_hard_boundary, unwrap_replicated_int


def _slots(
    *,
    train: int = 0,
    rollout: int = 0,
    updates: int = 4,
    max_lag: int = 0,
    inflight: int = 0,
    ready: int = 0,
    max_inflight: int = 8,
    trained: int = 0,
    total: int = 20,
    boundary: int = 20,
) -> int:
    return launch_slots(
        train_version=train,
        rollout_version=rollout,
        num_updates_per_batch=updates,
        max_policy_lag=max_lag,
        inflight_count=inflight,
        ready_count=ready,
        max_inflight=max_inflight,
        trained_batches=trained,
        num_rollouts=total,
        hard_boundary=boundary,
    )


def test_launch_slots_uses_optimizer_update_lag() -> None:
    assert _slots(max_lag=0) == 1
    assert _slots(max_lag=4) == 2
    assert _slots(train=4, max_lag=4) == 1
    assert _slots(train=8, max_lag=4) == 0


def test_launch_slots_subtracts_all_outstanding_batches() -> None:
    assert _slots(max_lag=8, inflight=1, ready=1) == 1
    assert _slots(max_lag=8, inflight=2, ready=1) == 0


def test_launch_slots_respects_concurrency_target_and_hard_boundary() -> None:
    assert _slots(max_lag=100, max_inflight=2) == 2
    assert _slots(max_lag=100, inflight=1, max_inflight=2) == 1
    assert _slots(max_lag=100, trained=3, boundary=4) == 1
    assert _slots(max_lag=100, trained=4, boundary=4) == 0


@pytest.mark.parametrize(
    ("updates", "max_lag", "expected_batches"),
    [
        (4, 0, 1),
        (4, 12, 4),
        (2, 22, 12),
    ],
)
def test_single_thread_cycle_exhausts_queue_before_sync(
    updates: int,
    max_lag: int,
    expected_batches: int,
) -> None:
    train = 0
    rollout = 0
    ready = 0
    consumed = 0
    while True:
        ready += _slots(
            train=train,
            rollout=rollout,
            updates=updates,
            max_lag=max_lag,
            ready=ready,
            max_inflight=64,
            trained=consumed,
            total=100,
            boundary=100,
        )
        if ready == 0:
            break
        ready -= 1
        consumed += 1
        train += updates

    assert consumed == expected_batches
    assert ready == 0


def test_launch_slots_rejects_future_rollout_version() -> None:
    with pytest.raises(ValueError, match="ahead of train_version"):
        _slots(train=2, rollout=3)


def test_policy_versions_track_train_updates_and_rollout_sync_separately() -> None:
    state = PolicyVersionState()
    state.record_optimizer_updates(4)
    assert state.train_version == 4
    assert state.rollout_version == 0
    assert state.rollout_lag == 4

    state.mark_rollout_synced(4)
    assert state.rollout_version == 4
    state.mark_rollout_synced(4)
    assert state.train_version == 4
    assert state.rollout_version == 4


def test_policy_versions_fail_closed_on_future_batch_or_sync() -> None:
    state = PolicyVersionState(train_version=3, rollout_version=2)
    with pytest.raises(ValueError, match="future behavior version"):
        state.behavior_lag(4)
    with pytest.raises(ValueError, match="future train version"):
        state.mark_rollout_synced(4)


def test_next_hard_boundary_uses_nearest_eval_save_or_final() -> None:
    assert next_hard_boundary(0, num_rollouts=20, eval_interval=4, save_interval=3) == 3
    assert next_hard_boundary(3, num_rollouts=20, eval_interval=4, save_interval=3) == 4
    assert next_hard_boundary(19, num_rollouts=20, eval_interval=4, save_interval=3) == 20


def test_train_step_result_aggregates_only_committed_optimizer_updates() -> None:
    committed = TrainStepResult(1.0, 1.0, 1e-6, True, [], {}, optimizer_updates=1)
    skipped = TrainStepResult(float("nan"), float("nan"), 1e-6, True, [], {}, optimizer_updates=0)
    result = _aggregate_update_results([committed, skipped])
    assert result.has_backward is True
    assert result.optimizer_updates == 1


def test_replicated_optimizer_count_must_agree() -> None:
    assert unwrap_replicated_int([7, 7], name="optimizer count") == 7
    with pytest.raises(RuntimeError, match="disagrees across workers"):
        unwrap_replicated_int([7, 8], name="optimizer count")
