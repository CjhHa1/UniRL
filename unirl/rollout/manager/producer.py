"""Independent rollout producer built on the shared :class:`RolloutManager`."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Deque, List, Optional, Tuple

from unirl.rollout.manager.admission import boundary_launch_slots, next_hard_boundary

if TYPE_CHECKING:
    from unirl.rollout.manager.rollout import RolloutManager
    from unirl.types.sample import Sample


BuildSample = Callable[[int], "Sample"]
RolloutGroups = List[List["Sample"]]


class ProducerState(str, Enum):
    INIT = "init"
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    QUIESCING = "quiescing"
    PAUSED = "paused"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class ProducerSnapshot:
    state: ProducerState
    inflight: int
    completed_queue: int
    source_cursor: int
    consumer_cursor: int
    launched: int
    completed: int
    consumed: int
    producer_wait_s: float
    pause_reason: str


class ContinuousRolloutProducer:
    """Keep a bounded rollout queue full while the main thread scores and trains.

    The producer thread is the sole active caller of ``RolloutManager``. The
    training thread may publish weights only after ``pause_and_drain`` transfers
    quiescent ownership to it.
    """

    _POLL_S = 0.01

    def __init__(
        self,
        manager: "RolloutManager",
        *,
        build_sample: BuildSample,
        batch_size: int,
        num_rollouts: int,
        start_rollout: int,
        current_version: int,
        max_inflight: int,
        max_pending_generations: int,
        weight_sync_interval: int,
        save_interval: int,
        eval_interval: int,
        timeout_s: float,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 0 <= int(start_rollout) <= int(num_rollouts):
            raise ValueError(f"start_rollout must be in [0, num_rollouts], got {start_rollout} of {num_rollouts}")
        if int(current_version) < 0:
            raise ValueError(f"current_version must be >= 0, got {current_version}")
        if int(max_inflight) < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight}")
        if int(max_pending_generations) < int(max_inflight):
            raise ValueError(
                f"max_pending_generations must be >= max_inflight, got {max_pending_generations} < {max_inflight}"
            )
        if int(weight_sync_interval) < 1:
            raise ValueError(f"weight_sync_interval must be >= 1, got {weight_sync_interval}")
        if int(save_interval) < 0 or int(eval_interval) < 0:
            raise ValueError(f"save/eval intervals must be >= 0, got {save_interval}/{eval_interval}")
        if float(timeout_s) <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")

        self._manager = manager
        self._build_sample = build_sample
        self._batch_size = int(batch_size)
        self._num_rollouts = int(num_rollouts)
        self._source_cursor = int(start_rollout)
        self._consumer_cursor = int(start_rollout)
        self._current_version = int(current_version)
        self._max_inflight = int(max_inflight)
        self._max_pending = int(max_pending_generations)
        self._weight_sync_interval = int(weight_sync_interval)
        self._save_interval = int(save_interval)
        self._eval_interval = int(eval_interval)
        self._timeout_s = float(timeout_s)

        self._cv = threading.Condition()
        self._queue: Deque[RolloutGroups] = deque()
        self._carried: List["Sample"] = []
        self._state = ProducerState.INIT
        self._pause_requested = False
        self._quiesce_requested = False
        self._stop_requested = False
        self._pause_reason = ""
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._inflight = 0
        self._batches_since_sync = 0
        self._launched = 0
        self._completed = 0
        self._consumed = 0
        self._producer_wait_s = 0.0

    def start(self) -> None:
        with self._cv:
            if self._state is not ProducerState.INIT:
                raise RuntimeError(f"producer can only start from INIT, got {self._state.value}")
            self._state = ProducerState.RUNNING
            self._thread = threading.Thread(target=self._run, name="unirl-rollout-producer", daemon=True)
            self._thread.start()

    @property
    def thread_alive(self) -> bool:
        with self._cv:
            thread = self._thread
        return thread is not None and thread.is_alive()

    def snapshot(self) -> ProducerSnapshot:
        with self._cv:
            return ProducerSnapshot(
                state=self._state,
                inflight=self._inflight,
                completed_queue=len(self._queue),
                source_cursor=self._source_cursor,
                consumer_cursor=self._consumer_cursor,
                launched=self._launched,
                completed=self._completed,
                consumed=self._consumed,
                producer_wait_s=self._producer_wait_s,
                pause_reason=self._pause_reason,
            )

    def take_next(self, *, timeout_s: Optional[float] = None) -> Tuple[RolloutGroups, float]:
        timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        with self._cv:
            self._raise_if_failed_locked()
            while not self._queue:
                self._raise_if_failed_locked()
                if self._state is ProducerState.STOPPED:
                    raise RuntimeError("rollout producer stopped before producing the requested batch")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for rollout production "
                        f"(state={self._state.value}, inflight={self._inflight})"
                    )
                self._cv.wait(timeout=remaining)

            hard_boundary = self._next_hard_boundary_locked()
            groups = self._queue.popleft()
            self._consumer_cursor += 1
            self._batches_since_sync += 1
            self._consumed += 1

            reasons = []
            if self._batches_since_sync >= self._weight_sync_interval:
                reasons.append("weight_sync")
            if self._consumer_cursor == hard_boundary:
                reasons.append("hard_boundary")
            if reasons:
                self._request_pause_locked("+".join(reasons))
            self._cv.notify_all()
            return groups, time.monotonic() - started

    def set_current_version(self, current_version: int) -> None:
        with self._cv:
            self._raise_if_failed_locked()
            current_version = int(current_version)
            if current_version < self._current_version:
                raise ValueError(f"current_version cannot move backwards: {current_version} < {self._current_version}")
            self._current_version = current_version

    def pause_and_drain(
        self,
        reason: str,
        *,
        require_empty: bool = False,
        timeout_s: Optional[float] = None,
    ) -> None:
        timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout
        with self._cv:
            self._raise_if_failed_locked()
            if self._state is ProducerState.STOPPED:
                raise RuntimeError("cannot pause a stopped rollout producer")
            if self._state is not ProducerState.PAUSED:
                self._request_pause_locked(str(reason))
                self._quiesce_requested = True
                self._cv.notify_all()
            while self._state is not ProducerState.PAUSED:
                self._raise_if_failed_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out quiescing rollout producer (state={self._state.value}, inflight={self._inflight})"
                    )
                self._cv.wait(timeout=remaining)
            if self._inflight:
                raise AssertionError(f"producer acknowledged PAUSED with inflight={self._inflight}")
            if require_empty and (self._queue or self._carried or not self._manager.empty):
                raise RuntimeError(
                    "durable rollout boundary requires an empty producer "
                    f"(queue={len(self._queue)}, carried={len(self._carried)})"
                )
            if require_empty and self._source_cursor != self._consumer_cursor:
                raise RuntimeError(
                    "durable rollout boundary requires matching source and consumer cursors "
                    f"({self._source_cursor} != {self._consumer_cursor})"
                )

    def publish(self, weight_sync: object, *, output_version: int) -> int:
        with self._cv:
            self._raise_if_failed_locked()
            if self._state is not ProducerState.PAUSED or self._inflight:
                raise RuntimeError(
                    f"weight publication requires a paused producer; "
                    f"state={self._state.value}, inflight={self._inflight}"
                )
            published = self._manager.sync_weights(weight_sync, output_version=output_version)
            self._current_version = published
            self._batches_since_sync = 0
            return published

    def resume(self, *, current_version: int, reset_sync_window: bool = False) -> None:
        with self._cv:
            self._raise_if_failed_locked()
            if self._state is not ProducerState.PAUSED:
                raise RuntimeError(f"producer can only resume from PAUSED, got {self._state.value}")
            current_version = int(current_version)
            if current_version < self._current_version:
                raise ValueError(f"current_version cannot move backwards: {current_version} < {self._current_version}")
            self._current_version = current_version
            if reset_sync_window:
                self._batches_since_sync = 0
            if self._carried:
                self._manager.submit(self._carried)
                self._inflight += len(self._carried)
                self._carried = []
            self._pause_requested = False
            self._quiesce_requested = False
            self._pause_reason = ""
            self._state = ProducerState.RUNNING
            self._cv.notify_all()

    def stop(self, *, timeout_s: Optional[float] = None, raise_on_error: bool = True) -> None:
        timeout = min(self._timeout_s, 30.0) if timeout_s is None else float(timeout_s)
        with self._cv:
            if self._state is ProducerState.INIT:
                self._state = ProducerState.STOPPED
                return
            self._stop_requested = True
            self._pause_requested = True
            self._quiesce_requested = True
            self._cv.notify_all()
            thread = self._thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout + self._POLL_S)
            if thread.is_alive():
                raise TimeoutError(
                    f"timed out stopping rollout producer "
                    f"(state={self.snapshot().state.value}, inflight={self.snapshot().inflight})"
                )

        with self._cv:
            self._queue.clear()
            self._carried = []
            if raise_on_error:
                self._raise_if_failed_locked()

    def _run(self) -> None:
        try:
            while True:
                with self._cv:
                    if self._stop_requested or self._quiesce_requested:
                        self._state = ProducerState.QUIESCING
                        current_version = self._current_version
                        should_quiesce = True
                        launch_ids = []
                    elif self._pause_requested:
                        self._state = ProducerState.PAUSE_REQUESTED
                        started = time.monotonic()
                        self._cv.wait()
                        self._producer_wait_s += time.monotonic() - started
                        continue
                    else:
                        should_quiesce = False
                        slots = self._launch_slots_locked()
                        launch_ids = list(range(self._source_cursor, self._source_cursor + slots))
                        if launch_ids:
                            self._source_cursor += len(launch_ids)
                            self._inflight += len(launch_ids)
                            self._launched += len(launch_ids)
                            self._assert_capacity_locked()
                            self._cv.notify_all()
                        current_version = self._current_version

                if launch_ids:
                    for gen_id in launch_ids:
                        self._manager.submit([self._build_sample(gen_id)])
                    continue

                if should_quiesce:
                    carried = self._manager.quiesce(current_version=current_version)
                    with self._cv:
                        self._inflight = 0
                    completed = self._drain_ready(current_version=current_version)
                    with self._cv:
                        self._carried = carried
                        if self._stop_requested:
                            self._queue.clear()
                            self._carried = []
                            self._state = ProducerState.STOPPED
                            self._cv.notify_all()
                            return
                        self._completed += completed
                        self._state = ProducerState.PAUSED
                        self._cv.notify_all()
                        while self._state is ProducerState.PAUSED and not self._stop_requested:
                            self._cv.wait()
                        if self._stop_requested:
                            self._queue.clear()
                            self._carried = []
                            self._state = ProducerState.STOPPED
                            self._cv.notify_all()
                            return
                    continue

                groups = self._manager.collect_ready(self._batch_size, current_version=current_version)
                if groups is None:
                    inflight_count, _ = self._manager.counts
                    if inflight_count == 0:
                        manager_empty = self._manager.empty
                        with self._cv:
                            tracked_inflight = self._inflight
                        if tracked_inflight and manager_empty:
                            raise RuntimeError(
                                f"{tracked_inflight} rollout generation(s) resolved without a complete batch"
                            )
                        if tracked_inflight:
                            groups = self._manager.collect(self._batch_size, current_version=current_version)
                    if groups is None:
                        with self._cv:
                            started = time.monotonic()
                            self._cv.wait(timeout=self._POLL_S)
                            self._producer_wait_s += time.monotonic() - started
                            continue

                with self._cv:
                    self._queue.append(groups)
                    self._inflight -= 1
                    self._completed += 1
                    self._assert_capacity_locked()
                    self._cv.notify_all()
        except BaseException as exc:
            with self._cv:
                self._error = exc
                self._state = ProducerState.FAILED
                self._cv.notify_all()

    def _drain_ready(self, *, current_version: int) -> int:
        completed = 0
        while True:
            groups = self._manager.collect_ready(self._batch_size, current_version=current_version)
            if groups is None:
                return completed
            with self._cv:
                self._queue.append(groups)
                self._assert_capacity_locked()
                self._cv.notify_all()
            completed += 1

    def _launch_slots_locked(self) -> int:
        hard_boundary = self._next_hard_boundary_locked()
        return boundary_launch_slots(
            inflight_count=self._inflight,
            ready_count=len(self._queue),
            max_inflight=self._max_inflight,
            max_pending=self._max_pending,
            trained_batches=self._consumer_cursor,
            num_rollouts=self._num_rollouts,
            hard_boundary=hard_boundary,
            batches_since_sync=self._batches_since_sync,
            weight_sync_interval=self._weight_sync_interval,
        )

    def _next_hard_boundary_locked(self) -> int:
        return next_hard_boundary(
            self._consumer_cursor,
            num_rollouts=self._num_rollouts,
            eval_interval=self._eval_interval,
            save_interval=self._save_interval,
        )

    def _request_pause_locked(self, reason: str) -> None:
        if reason:
            reasons = set(self._pause_reason.split("+")) if self._pause_reason else set()
            reasons.update(reason.split("+"))
            self._pause_reason = "+".join(sorted(reasons))
        self._pause_requested = True
        if self._state is ProducerState.RUNNING:
            self._state = ProducerState.PAUSE_REQUESTED
        self._cv.notify_all()

    def _assert_capacity_locked(self) -> None:
        outstanding = self._inflight + len(self._queue)
        if outstanding > self._max_pending:
            raise AssertionError(f"rollout outstanding capacity exceeded: {outstanding} > {self._max_pending}")

    def _raise_if_failed_locked(self) -> None:
        if self._state is ProducerState.FAILED:
            if self._error is None:
                raise RuntimeError("rollout producer failed without an exception")
            raise RuntimeError("rollout producer failed") from self._error


__all__ = [
    "ContinuousRolloutProducer",
    "ProducerSnapshot",
    "ProducerState",
]
