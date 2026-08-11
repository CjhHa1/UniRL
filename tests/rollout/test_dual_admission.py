from __future__ import annotations

import threading
import time
from typing import Callable

import pytest

from unirl.rollout.manager import RolloutManager
from unirl.trainer.async_rollout import AsyncRolloutTrainerMixin, boundary_launch_slots, next_hard_boundary
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def _request(gen_id: int) -> Sample:
    return Sample(
        parts=[
            Part.input(
                [f"root-{gen_id}"],
                metadata=[{"rollout_id": gen_id}],
            )
        ]
    )


def _completed(request: Sample, *, output_version: int = 0) -> Sample:
    root = request.parts[0]
    generated = Part(
        sample_ids=[f"{root.sample_ids[0]}/0"],
        sampling_params=ARSamplingParams(),
        output_version=output_version,
    )
    return Sample(parts=[root, generated])


class _Rollout:
    def __init__(self) -> None:
        self.version = 0
        self.stopping = False

    def set_stopping(self, stopping: bool) -> None:
        self.stopping = stopping

    def set_version(self, version: int) -> None:
        self.version = version


class _Pending:
    def __init__(self, value: Sample) -> None:
        self._value = value
        self._ready = threading.Event()

    def complete(self) -> None:
        self._ready.set()

    def ready(self) -> bool:
        return self._ready.is_set()

    def result(self) -> Sample:
        return self._value


def test_dual_admission_uses_pool_queue_without_crossing_sync_window() -> None:
    assert (
        boundary_launch_slots(
            inflight_count=0,
            ready_count=0,
            max_inflight=1,
            max_pending=2,
            leased_count=0,
            trained_batches=0,
            num_rollouts=8,
            hard_boundary=8,
            batches_since_sync=0,
            weight_sync_interval=4,
        )
        == 2
    )
    assert (
        boundary_launch_slots(
            inflight_count=1,
            ready_count=0,
            max_inflight=1,
            max_pending=2,
            leased_count=1,
            trained_batches=0,
            num_rollouts=8,
            hard_boundary=8,
            batches_since_sync=0,
            weight_sync_interval=4,
        )
        == 1
    )
    assert (
        boundary_launch_slots(
            inflight_count=0,
            ready_count=0,
            max_inflight=1,
            max_pending=2,
            leased_count=1,
            trained_batches=3,
            num_rollouts=8,
            hard_boundary=8,
            batches_since_sync=3,
            weight_sync_interval=4,
        )
        == 0
    )


@pytest.mark.parametrize(
    ("num_rollouts", "max_inflight", "max_pending", "sync_interval", "save_interval", "eval_interval"),
    [
        (7, 1, 1, 1, 0, 0),
        (11, 1, 3, 4, 0, 0),
        (13, 2, 4, 3, 0, 0),
        (9, 3, 5, 4, 3, 0),
        (12, 4, 8, 5, 0, 4),
        (15, 4, 8, 6, 5, 4),
    ],
)
def test_dual_admission_stays_bounded_across_all_boundaries(
    num_rollouts: int,
    max_inflight: int,
    max_pending: int,
    sync_interval: int,
    save_interval: int,
    eval_interval: int,
) -> None:
    consumer_cursor = 0
    source_cursor = 0
    manager_outstanding = 0
    batches_since_sync = 0

    while consumer_cursor < num_rollouts:
        hard_boundary = next_hard_boundary(
            consumer_cursor,
            num_rollouts=num_rollouts,
            eval_interval=eval_interval,
            save_interval=save_interval,
        )
        slots = boundary_launch_slots(
            inflight_count=manager_outstanding,
            ready_count=0,
            max_inflight=max_inflight,
            max_pending=max_pending,
            trained_batches=consumer_cursor,
            num_rollouts=num_rollouts,
            hard_boundary=hard_boundary,
            batches_since_sync=batches_since_sync,
            weight_sync_interval=sync_interval,
        )
        manager_outstanding += slots
        source_cursor += slots
        assert 0 < manager_outstanding <= max_pending
        assert source_cursor <= hard_boundary

        manager_outstanding -= 1
        slots = boundary_launch_slots(
            inflight_count=manager_outstanding,
            ready_count=0,
            max_inflight=max_inflight,
            max_pending=max_pending,
            leased_count=1,
            trained_batches=consumer_cursor,
            num_rollouts=num_rollouts,
            hard_boundary=hard_boundary,
            batches_since_sync=batches_since_sync,
            weight_sync_interval=sync_interval,
        )
        manager_outstanding += slots
        source_cursor += slots
        assert manager_outstanding <= max_pending
        assert source_cursor <= hard_boundary

        consumer_cursor += 1
        batches_since_sync += 1
        boundary_due = (
            consumer_cursor == hard_boundary or batches_since_sync >= sync_interval or consumer_cursor == num_rollouts
        )
        if boundary_due:
            assert manager_outstanding == 0
            batches_since_sync = 0

    assert source_cursor == consumer_cursor == num_rollouts


def test_rollout_pool_launches_prequeued_generation_without_collection() -> None:
    rollout = _Rollout()
    launched: list[_Pending] = []
    lock = threading.Lock()

    def launch(request: Sample) -> _Pending:
        pending = _Pending(_completed(request))
        with lock:
            launched.append(pending)
        return pending

    manager = RolloutManager(rollout, launchers=[launch], capacities=[1], group_size=1)
    manager.submit([_request(0), _request(1)])
    try:
        _wait_until(lambda: len(launched) == 1)
        launched[0].complete()
        _wait_until(lambda: len(launched) == 2)

        first = manager.collect(1, current_version=0)
        assert first[0][0].parts[0].metadata == [{"rollout_id": 0}]

        launched[1].complete()
        second = manager.collect(1, current_version=0)
        assert second[0][0].parts[0].metadata == [{"rollout_id": 1}]
    finally:
        manager.close()


def test_dual_refills_before_scoring() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.count_reads = 0

        @property
        def counts(self) -> tuple[int, int]:
            self.count_reads += 1
            return (0, 0) if self.count_reads == 1 else (1, 0)

        def collect(self, n: int, *, current_version: int) -> list[list[Sample]]:
            assert n == 1
            assert current_version == 0
            return [[_completed(_request(0))]]

    trainer = AsyncRolloutTrainerMixin()
    trainer._rollout_manager = _Manager()
    trainer.batch_size = 1
    trainer._max_inflight = 1
    trainer._weight_sync_interval = 4
    trainer._batches_since_sync = 0
    trainer._train_version = 0
    submitted: list[int] = []
    trainer._submit_generations = submitted.append

    def score(gen_id: int, completed: Sample) -> Sample:
        assert gen_id == 0
        assert submitted == [2, 1]
        return completed

    trainer._score_completed = score
    trainer._next_rollout_batch(
        0,
        num_rollouts=8,
        hard_boundary=8,
        max_pending=2,
    )
