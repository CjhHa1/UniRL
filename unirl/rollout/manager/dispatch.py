from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Deque, List, Optional, Sequence

if TYPE_CHECKING:
    from unirl.types.sample import Sample


Launch = Callable[["Sample"], Any]


@dataclass(frozen=True)
class _PendingUnit:
    sequence: int
    launcher: int
    task: "Sample"
    pending: Any


class RolloutPool:
    _PROBE_INTERVAL_S = 0.01

    def __init__(
        self,
        launchers: Sequence[Launch],
        capacities: Sequence[int],
        *,
        worker_max_concurrency: int = 0,
    ) -> None:
        self._launchers = list(launchers)
        self._capacities = [int(capacity) for capacity in capacities]
        if not self._launchers:
            raise ValueError("RolloutPool requires at least one launcher")
        if len(self._launchers) != len(self._capacities):
            raise ValueError(
                f"RolloutPool launcher/capacity count mismatch: {len(self._launchers)} != {len(self._capacities)}"
            )
        if any(capacity <= 0 for capacity in self._capacities):
            raise ValueError(f"RolloutPool capacities must be positive; got {self._capacities}")
        if worker_max_concurrency:
            required = max(self._capacities) + 2
            if worker_max_concurrency < required:
                raise ValueError(
                    f"worker_max_concurrency ({worker_max_concurrency}) must be >= launcher capacity + 2 ({required})"
                )

        self._queue: Deque[tuple[int, "Sample"]] = deque()
        self._running: List[_PendingUnit] = []
        self._completed: Deque[_PendingUnit] = deque()
        self._resolving: List[_PendingUnit] = []
        self._launching: Optional[tuple[int, int, "Sample"]] = None
        self._next_sequence = 0
        self._next_launcher = 0
        self._paused = True
        self._closed = False
        self._failure: Optional[BaseException] = None
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._progress, name="rollout-pool", daemon=True)
        self._thread.start()

    def add(self, tasks: List["Sample"]) -> None:
        with self._condition:
            self._raise_if_unavailable()
            for task in tasks:
                self._queue.append((self._next_sequence, task))
                self._next_sequence += 1
            self._paused = False
            self._condition.notify_all()

    def pause(self) -> List["Sample"]:
        with self._condition:
            if self._closed:
                raise RuntimeError("RolloutPool is closed")
            self._paused = True
            tasks = [task for _, task in self._queue]
            self._queue.clear()
            self._condition.notify_all()
            return tasks

    def take_completed(self, *, block: bool) -> List[_PendingUnit]:
        with self._condition:
            while block and not self._completed and self._has_remote_work() and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            completed = list(self._completed)
            self._completed.clear()
            self._resolving.extend(completed)
            return completed

    def drain(self) -> List[_PendingUnit]:
        with self._condition:
            while self._launching is not None or self._running:
                self._condition.wait()
            completed = list(self._completed)
            self._completed.clear()
            self._resolving.extend(completed)
            return completed

    def acknowledge(self, units: List[_PendingUnit]) -> None:
        with self._condition:
            for unit in units:
                if unit in self._resolving:
                    self._resolving.remove(unit)
            self._condition.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self._condition:
            self._record_failure_locked(exc)

    @property
    def live(self) -> bool:
        with self._condition:
            self._raise_if_failed()
            return self._has_work()

    @property
    def has_work(self) -> bool:
        with self._condition:
            return self._has_work()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._paused = True
            self._queue.clear()
            self._condition.notify_all()
        self._thread.join()

    def _has_remote_work(self) -> bool:
        return bool(self._queue or self._launching is not None or self._running)

    def _has_work(self) -> bool:
        return bool(self._queue or self._launching is not None or self._running or self._completed or self._resolving)

    def _raise_if_unavailable(self) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("RolloutPool is closed")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _progress(self) -> None:
        while True:
            with self._condition:
                if self._closed and self._launching is None and not self._running:
                    return
                reservation = self._reserve_launch()
                if reservation is not None:
                    running = []
                else:
                    running = list(self._running)
                if reservation is None and not running:
                    self._condition.wait()
                    continue

            if reservation is not None:
                sequence, launcher_index, task = reservation
                try:
                    pending = self._launchers[launcher_index](task)
                except BaseException as exc:
                    with self._condition:
                        self._launching = None
                        if not self._closed:
                            self._queue.appendleft((sequence, task))
                        self._record_failure_locked(exc)
                    continue
                with self._condition:
                    self._launching = None
                    self._running.append(_PendingUnit(sequence, launcher_index, task, pending))
                    self._condition.notify_all()
                continue

            ready = []
            probe_error: Optional[BaseException] = None
            for unit in running:
                try:
                    if unit.pending.ready():
                        ready.append(unit)
                except BaseException as exc:
                    if probe_error is None:
                        probe_error = exc
                    ready.append(unit)
            if not ready:
                with self._condition:
                    self._condition.wait(timeout=self._PROBE_INTERVAL_S)
                continue

            with self._condition:
                if probe_error is not None:
                    self._record_failure_locked(probe_error)
                for unit in ready:
                    if unit not in self._running:
                        continue
                    self._running.remove(unit)
                    self._completed.append(unit)
                self._condition.notify_all()

    def _reserve_launch(self) -> Optional[tuple[int, int, "Sample"]]:
        if self._paused or self._closed or self._failure is not None or not self._queue:
            return None
        load = [0] * len(self._launchers)
        for unit in [*self._running, *self._completed, *self._resolving]:
            load[unit.launcher] += 1
        if self._launching is not None:
            load[self._launching[1]] += 1
        for offset in range(len(self._launchers)):
            index = (self._next_launcher + offset) % len(self._launchers)
            if load[index] < self._capacities[index]:
                sequence, task = self._queue.popleft()
                reservation = (sequence, index, task)
                self._launching = reservation
                self._next_launcher = (index + 1) % len(self._launchers)
                return reservation
        return None

    def _record_failure_locked(self, exc: BaseException) -> None:
        if self._failure is None:
            self._failure = exc
        self._paused = True
        self._condition.notify_all()


__all__ = ["RolloutPool"]
