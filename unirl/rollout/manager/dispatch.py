from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Deque, List, Optional

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Slot
    from unirl.types.sample import Sample


@dataclass(frozen=True)
class _Running:
    slot: int
    pending: Any


class TrajectoryPool:
    _PROBE_INTERVAL_S = 0.01

    def __init__(
        self,
        slots: List["Slot"],
        *,
        per_slot_inflight: int,
        worker_max_concurrency: int = 0,
    ) -> None:
        self._slots = list(slots)
        self._cap = int(per_slot_inflight)
        if not self._slots:
            raise ValueError("TrajectoryPool requires at least one engine slot")
        if self._cap <= 0:
            raise ValueError(f"per_slot_inflight must be positive; got {self._cap}")
        if worker_max_concurrency and worker_max_concurrency < self._cap + 2:
            raise ValueError(
                f"worker_max_concurrency ({worker_max_concurrency}) must be >= per_slot_inflight + 2 ({self._cap + 2})"
            )
        self._queue: Deque["Sample"] = deque()
        self._running: List[_Running] = []
        self._completed: Deque["Sample"] = deque()
        self._paused = True
        self._closed = False
        self._failure: Optional[BaseException] = None
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._progress, name="trajectory-pool", daemon=True)
        self._thread.start()

    def add(self, tasks: List["Sample"]) -> None:
        with self._condition:
            self._raise_if_unavailable()
            self._queue.extend(tasks)
            self._paused = False
            self._condition.notify_all()

    def pause(self) -> List["Sample"]:
        with self._condition:
            self._raise_if_failed()
            self._paused = True
            tasks = list(self._queue)
            self._queue.clear()
            self._condition.notify_all()
            return tasks

    def take_completed(self, *, block: bool) -> List["Sample"]:
        with self._condition:
            while block and not self._completed and self._has_remote_work() and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            completed = list(self._completed)
            self._completed.clear()
            return completed

    def drain(self) -> List["Sample"]:
        with self._condition:
            while self._running and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            completed = list(self._completed)
            self._completed.clear()
            return completed

    @property
    def live(self) -> bool:
        with self._condition:
            self._raise_if_failed()
            return bool(self._queue or self._running or self._completed)

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
        return bool(self._queue or self._running)

    def _raise_if_unavailable(self) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("TrajectoryPool is closed")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _progress(self) -> None:
        while True:
            with self._condition:
                if self._closed and not self._running:
                    return
                if self._failure is not None:
                    return
                try:
                    self._launch_to_capacity()
                except BaseException as exc:
                    self._failure = exc
                    self._condition.notify_all()
                    return
                running = list(self._running)
                if not running:
                    self._condition.wait()
                    continue

            try:
                ready = [item for item in running if item.pending.ready()]
            except BaseException as exc:
                self._record_failure(exc)
                return
            if not ready:
                with self._condition:
                    self._condition.wait(timeout=self._PROBE_INTERVAL_S)
                continue

            for item in ready:
                try:
                    result = item.pending.result()
                except BaseException as exc:
                    self._record_failure(exc)
                    return
                with self._condition:
                    if item not in self._running:
                        continue
                    self._running.remove(item)
                    self._completed.append(result)
                    self._condition.notify_all()

    def _launch_to_capacity(self) -> None:
        if self._paused or self._closed:
            return
        load = [0] * len(self._slots)
        for item in self._running:
            load[item.slot] += 1
        for index, slot in enumerate(self._slots):
            while self._queue and load[index] < self._cap:
                pending = slot.launch("run_trajectory", self._queue.popleft())
                self._running.append(_Running(index, pending))
                load[index] += 1

    def _record_failure(self, exc: BaseException) -> None:
        with self._condition:
            self._failure = exc
            self._paused = True
            self._condition.notify_all()


__all__ = ["TrajectoryPool"]
