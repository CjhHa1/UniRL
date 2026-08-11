from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import pytest

from unirl.rollout.manager import RolloutManager
from unirl.rollout.manager.producer import ContinuousRolloutProducer, ProducerState
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


class _WeightSync:
    def __init__(self) -> None:
        self.calls = 0

    def sync(self) -> None:
        self.calls += 1


class _Pending:
    def __init__(self, value: Sample) -> None:
        self._value = value

    def ready(self) -> bool:
        return True

    def result(self) -> Sample:
        return self._value


class _Rollout:
    def __init__(self) -> None:
        self.version = 0
        self.stopping = False

    def set_stopping(self, stopping: bool) -> None:
        self.stopping = stopping

    def set_version(self, version: int) -> None:
        self.version = version


class _ImmediateManager:
    """Minimal thread-safe RolloutManager stand-in with immediate completions."""

    def __init__(self, *, fail_collect: bool = False) -> None:
        self._ready: deque[int] = deque()
        self._lock = threading.Lock()
        self._fail_collect = fail_collect
        self.published_version = 0
        self.quiesce_versions: list[int] = []

    def submit(self, tasks: list[int]) -> None:
        with self._lock:
            self._ready.extend(tasks)

    def collect_ready(self, n: int, *, current_version: int) -> list[list[int]] | None:
        del current_version
        with self._lock:
            if self._fail_collect:
                raise RuntimeError("collect failed")
            if len(self._ready) < n:
                return None
            return [[self._ready.popleft()] for _ in range(n)]

    def collect(self, n: int, *, current_version: int) -> list[list[int]]:
        groups = self.collect_ready(n, current_version=current_version)
        if groups is None:
            raise RuntimeError("rollout underflow")
        return groups

    def quiesce(self, *, current_version: int) -> list[int]:
        self.quiesce_versions.append(current_version)
        return []

    def sync_weights(self, weight_sync: _WeightSync, *, output_version: int) -> int:
        with self._lock:
            if self._ready:
                raise RuntimeError("sync requires no rollout work")
            weight_sync.sync()
            self.published_version = output_version
            return output_version

    @property
    def counts(self) -> tuple[int, int]:
        with self._lock:
            return 0, len(self._ready)

    @property
    def empty(self) -> bool:
        with self._lock:
            return not self._ready


class _DroppingManager(_ImmediateManager):
    def submit(self, tasks: list[int]) -> None:
        del tasks


def _producer(
    manager: _ImmediateManager,
    *,
    num_rollouts: int,
    start_rollout: int = 0,
    max_inflight: int,
    max_pending: int,
    sync_interval: int = 100,
    save_interval: int = 0,
    eval_interval: int = 0,
) -> ContinuousRolloutProducer:
    return ContinuousRolloutProducer(
        manager,
        build_sample=lambda gen_id: gen_id,
        batch_size=1,
        num_rollouts=num_rollouts,
        start_rollout=start_rollout,
        current_version=0,
        max_inflight=max_inflight,
        max_pending_generations=max_pending,
        weight_sync_interval=sync_interval,
        save_interval=save_interval,
        eval_interval=eval_interval,
        timeout_s=5.0,
    )


def test_producer_refills_during_training_and_stays_bounded() -> None:
    producer = _producer(
        _ImmediateManager(),
        num_rollouts=6,
        max_inflight=2,
        max_pending=3,
    )
    producer.start()
    try:
        _wait_until(lambda: producer.snapshot().completed_queue == 3)
        initial = producer.snapshot()
        assert initial.launched == 3
        assert initial.inflight + initial.completed_queue <= 3

        groups, _ = producer.take_next()
        assert groups == [[0]]

        # No control call is needed while the consumer trains: the producer
        # refills the released queue slot independently.
        _wait_until(lambda: producer.snapshot().launched == 4)
        during_train = producer.snapshot()
        assert during_train.consumer_cursor == 1
        assert during_train.inflight + during_train.completed_queue <= 3
    finally:
        producer.stop()
    assert producer.snapshot().state is ProducerState.STOPPED


def test_resumed_producer_starts_from_committed_cursor() -> None:
    producer = _producer(
        _ImmediateManager(),
        num_rollouts=5,
        start_rollout=3,
        max_inflight=2,
        max_pending=2,
        sync_interval=4,
    )
    producer.start()
    try:
        _wait_until(lambda: producer.snapshot().completed_queue == 2)
        first, _ = producer.take_next()
        second, _ = producer.take_next()
        assert first == [[3]]
        assert second == [[4]]
        producer.pause_and_drain("final", require_empty=True)
    finally:
        producer.stop()


def test_producer_drives_rollout_manager_nonblocking_collection() -> None:
    rollout = _Rollout()

    def _completed(sample: Sample) -> _Pending:
        root = sample.parts[0]
        generated = Part(
            sample_ids=[f"{sample_id}/0" for sample_id in root.sample_ids],
            sampling_params=ARSamplingParams(),
            output_version=rollout.version,
        )
        return _Pending(Sample(parts=[root, generated]))

    manager = RolloutManager(
        rollout,
        launchers=[_completed],
        capacities=[1],
        group_size=1,
    )
    producer = ContinuousRolloutProducer(
        manager,
        build_sample=lambda gen_id: Sample(
            parts=[
                Part.input(
                    [f"root-{gen_id}-0", f"root-{gen_id}-1"],
                    metadata=[{"rollout_id": gen_id}, {"rollout_id": gen_id}],
                )
            ]
        ),
        batch_size=2,
        num_rollouts=1,
        start_rollout=0,
        current_version=0,
        max_inflight=1,
        max_pending_generations=1,
        weight_sync_interval=1,
        save_interval=0,
        eval_interval=0,
        timeout_s=5.0,
    )
    producer.start()
    try:
        groups, _ = producer.take_next()
        assert groups[0][0].parts[-1].output_version == 0
        producer.pause_and_drain("final", require_empty=True)
    finally:
        producer.stop()
        manager.close()


def test_hard_boundary_clamps_admission_and_transfers_quiescent_ownership() -> None:
    manager = _ImmediateManager()
    producer = _producer(
        manager,
        num_rollouts=4,
        max_inflight=4,
        max_pending=4,
        eval_interval=2,
    )
    sync = _WeightSync()
    producer.start()
    try:
        _wait_until(lambda: producer.snapshot().completed_queue == 2)
        assert producer.snapshot().source_cursor == 2

        producer.take_next()
        producer.take_next()
        producer.pause_and_drain("eval", require_empty=True)
        boundary = producer.snapshot()
        assert boundary.state is ProducerState.PAUSED
        assert boundary.inflight == 0
        assert boundary.completed_queue == 0
        assert boundary.source_cursor == 2
        assert boundary.consumer_cursor == 2

        assert producer.publish(sync, output_version=4) == 4
        assert sync.calls == 1
        producer.resume(current_version=4)
        _wait_until(lambda: producer.snapshot().completed_queue == 2)
        assert producer.snapshot().source_cursor == 4
    finally:
        producer.stop()


def test_sync_window_pauses_before_admitting_the_next_version() -> None:
    manager = _ImmediateManager()
    producer = _producer(
        manager,
        num_rollouts=4,
        max_inflight=4,
        max_pending=4,
        sync_interval=2,
    )
    sync = _WeightSync()
    producer.start()
    try:
        _wait_until(lambda: producer.snapshot().completed_queue == 2)
        assert producer.snapshot().source_cursor == 2
        producer.take_next()
        producer.take_next()
        producer.pause_and_drain("weight_sync", require_empty=True)
        assert producer.snapshot().source_cursor == 2

        producer.publish(sync, output_version=8)
        producer.resume(current_version=8)
        _wait_until(lambda: producer.snapshot().completed_queue == 2)
        assert producer.snapshot().source_cursor == 4
    finally:
        producer.stop()


def test_empty_optimizer_window_can_resume_without_weight_publication() -> None:
    producer = _producer(
        _ImmediateManager(),
        num_rollouts=4,
        max_inflight=2,
        max_pending=2,
        sync_interval=2,
    )
    producer.start()
    try:
        _wait_until(lambda: producer.snapshot().completed_queue == 2)
        producer.take_next()
        producer.take_next()
        producer.pause_and_drain("weight_sync", require_empty=True)

        # A batch may perform zero optimizer updates. The trainer then skips
        # weight publication but still starts a fresh cadence window.
        producer.resume(current_version=0, reset_sync_window=True)
        _wait_until(lambda: producer.snapshot().completed_queue == 2)
        assert producer.snapshot().source_cursor == 4
    finally:
        producer.stop()


def test_pause_waits_for_post_train_version_before_quiescing() -> None:
    manager = _ImmediateManager()
    producer = _producer(
        manager,
        num_rollouts=2,
        max_inflight=1,
        max_pending=1,
        sync_interval=1,
    )
    producer.start()
    try:
        _wait_until(lambda: producer.snapshot().completed_queue == 1)
        producer.take_next()
        _wait_until(lambda: producer.snapshot().state is ProducerState.PAUSE_REQUESTED)
        assert manager.quiesce_versions == []

        producer.set_current_version(4)
        producer.pause_and_drain("weight_sync", require_empty=True)
        assert manager.quiesce_versions == [4]
    finally:
        producer.stop()


def test_sample_build_does_not_hold_the_producer_condition() -> None:
    manager = _ImmediateManager()
    build_started = threading.Event()
    release_build = threading.Event()

    def _build(gen_id: int) -> int:
        build_started.set()
        assert release_build.wait(timeout=5.0)
        return gen_id

    producer = ContinuousRolloutProducer(
        manager,
        build_sample=_build,
        batch_size=1,
        num_rollouts=1,
        start_rollout=0,
        current_version=0,
        max_inflight=1,
        max_pending_generations=1,
        weight_sync_interval=1,
        save_interval=0,
        eval_interval=0,
        timeout_s=5.0,
    )
    producer.start()
    assert build_started.wait(timeout=5.0)

    snapshot_done = threading.Event()
    snapshot_thread = threading.Thread(
        target=lambda: (producer.snapshot(), snapshot_done.set()),
        daemon=True,
    )
    snapshot_thread.start()
    try:
        assert snapshot_done.wait(timeout=0.5)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out quiescing"):
            producer.pause_and_drain("test", timeout_s=0.05)
        assert time.monotonic() - started < 0.5
    finally:
        release_build.set()
        snapshot_thread.join(timeout=5.0)
        producer.stop()


def test_stop_timeout_does_not_report_a_live_thread_as_stopped() -> None:
    build_started = threading.Event()
    release_build = threading.Event()

    def _build(gen_id: int) -> int:
        build_started.set()
        assert release_build.wait(timeout=5.0)
        return gen_id

    producer = ContinuousRolloutProducer(
        _ImmediateManager(),
        build_sample=_build,
        batch_size=1,
        num_rollouts=1,
        start_rollout=0,
        current_version=0,
        max_inflight=1,
        max_pending_generations=1,
        weight_sync_interval=1,
        save_interval=0,
        eval_interval=0,
        timeout_s=5.0,
    )
    producer.start()
    assert build_started.wait(timeout=5.0)
    try:
        with pytest.raises(TimeoutError, match="timed out stopping"):
            producer.stop(timeout_s=0.01)
        assert producer.thread_alive
    finally:
        release_build.set()
        producer.stop(timeout_s=5.0)
    assert not producer.thread_alive


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
def test_state_machine_stays_bounded_across_all_boundaries(
    num_rollouts: int,
    max_inflight: int,
    max_pending: int,
    sync_interval: int,
    save_interval: int,
    eval_interval: int,
) -> None:
    manager = _ImmediateManager()
    producer = _producer(
        manager,
        num_rollouts=num_rollouts,
        max_inflight=max_inflight,
        max_pending=max_pending,
        sync_interval=sync_interval,
        save_interval=save_interval,
        eval_interval=eval_interval,
    )
    sync = _WeightSync()
    batches_since_sync = 0
    producer.start()
    try:
        for step in range(1, num_rollouts + 1):
            groups, _ = producer.take_next()
            assert groups == [[step - 1]]
            batches_since_sync += 1
            producer.set_current_version(step)

            eval_due = eval_interval > 0 and step % eval_interval == 0
            save_due = save_interval > 0 and (step % save_interval == 0 or step == num_rollouts)
            sync_due = step < num_rollouts and batches_since_sync >= sync_interval
            final_due = step == num_rollouts
            if eval_due or save_due or sync_due or final_due:
                producer.pause_and_drain(
                    "test-boundary",
                    require_empty=eval_due or save_due or final_due,
                )
                if eval_due or save_due or sync_due:
                    producer.publish(sync, output_version=step)
                    batches_since_sync = 0
                if not final_due:
                    producer.resume(current_version=step)

            snapshot = producer.snapshot()
            assert snapshot.inflight + snapshot.completed_queue <= max_pending
    finally:
        producer.stop()

    snapshot = producer.snapshot()
    assert snapshot.state is ProducerState.STOPPED
    assert snapshot.source_cursor == num_rollouts
    assert snapshot.consumer_cursor == num_rollouts
    assert snapshot.completed_queue == 0


def test_producer_failure_reaches_consumer() -> None:
    producer = _producer(
        _ImmediateManager(fail_collect=True),
        num_rollouts=1,
        max_inflight=1,
        max_pending=1,
    )
    producer.start()
    with pytest.raises(RuntimeError, match="rollout producer failed") as exc_info:
        producer.take_next()
    assert exc_info.value.__cause__ is not None
    assert str(exc_info.value.__cause__) == "collect failed"
    assert producer.snapshot().state is ProducerState.FAILED
    producer.stop(raise_on_error=False)


def test_filtered_generation_fails_instead_of_stranding_admission() -> None:
    producer = _producer(
        _DroppingManager(),
        num_rollouts=1,
        max_inflight=1,
        max_pending=1,
    )
    producer.start()
    with pytest.raises(RuntimeError, match="rollout producer failed") as exc_info:
        producer.take_next()
    assert exc_info.value.__cause__ is not None
    assert "resolved without a complete batch" in str(exc_info.value.__cause__)
    assert producer.snapshot().state is ProducerState.FAILED
    producer.stop(raise_on_error=False)


def test_invalid_pending_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_pending_generations must be >= max_inflight"):
        _producer(
            _ImmediateManager(),
            num_rollouts=1,
            max_inflight=2,
            max_pending=1,
        )
