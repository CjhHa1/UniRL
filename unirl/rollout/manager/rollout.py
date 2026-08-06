from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, List

from unirl.rollout.manager.buffers import CompletedGroups, PendingGroups, root_of
from unirl.rollout.manager.dispatch import TrajectoryPool
from unirl.rollout.manager.filters import RolloutFilter, identity

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Handle
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class RolloutUnderflow(RuntimeError):
    pass


class RolloutManager:
    def __init__(
        self,
        rollout: "Handle",
        *,
        group_size: int,
        per_worker_inflight: int,
        worker_max_concurrency: int = 0,
        filter_fn: RolloutFilter = identity,
    ) -> None:
        indices = [
            index
            for index, rank_info in enumerate(rollout.rank_infos)
            if rank_info.tp_rank == 0 and rank_info.pp_rank == 0
        ]
        self._rollout = rollout
        self._pool = TrajectoryPool(
            [rollout.slot(index) for index in indices],
            per_slot_inflight=per_worker_inflight,
            worker_max_concurrency=worker_max_concurrency,
        )
        self._pending = PendingGroups(group_size)
        self._buffer = CompletedGroups()
        self._filter = filter_fn
        self._weight_version = 0
        self._closed = False

    def submit(self, tasks: List["Sample"]) -> None:
        self._ensure_open()
        self._pool.add(list(tasks))

    def collect(self, n: int) -> List[List["Sample"]]:
        self._ensure_open()
        n = int(n)
        if n <= 0:
            raise ValueError(f"collect count must be positive; got {n}")
        selected = []
        while len(selected) < n:
            self._route(self._pool.take_completed(block=False), allow_suspended=False)
            while len(self._buffer) and len(selected) < n:
                kept = self._apply_filter(self._buffer.popleft())
                if kept:
                    selected.append(kept)
            if len(selected) == n:
                return selected
            if not self._pool.live:
                raise RolloutUnderflow(f"needed {n} rollout groups, collected {len(selected)}")
            self._route(self._pool.take_completed(block=True), allow_suspended=False)
        return selected

    def quiesce(self) -> List["Sample"]:
        self._ensure_open()
        undispatched = self._pool.pause()
        self._rollout.set_stopping(True)
        try:
            completed = self._pool.drain()
        finally:
            self._rollout.set_stopping(False)

        suspended = self._route(completed, allow_suspended=True)
        tails_by_root: Dict[str, List["Sample"]] = defaultdict(list)
        for sample in [*undispatched, *suspended]:
            tails_by_root[root_of(sample)].append(sample)

        carried = []
        for root, tails in tails_by_root.items():
            known = [*self._pending.get(root), *tails]
            if self._apply_filter(known):
                carried.extend(tails)
            else:
                discarded = self._pending.discard(root)
                logger.info(
                    "rollout filter discarded incomplete root=%s tails=%d completed=%d",
                    root,
                    len(tails),
                    discarded,
                )
        return carried

    def sync_weights(self, weight_sync: object) -> int:
        self._ensure_open()
        self._route(self._pool.take_completed(block=False), allow_suspended=False)
        if self._pool.live:
            raise RuntimeError("sync_weights requires no queued or in-flight trajectories")
        weight_sync.sync()
        self._weight_version += 1
        return self._weight_version

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._pool.live:
                self.quiesce()
        finally:
            self._pool.close()
            self._closed = True

    def _route(self, samples: List["Sample"], *, allow_suspended: bool) -> List["Sample"]:
        terminal = []
        suspended = []
        for sample in samples:
            status = sample.parts[-1].harness_status if sample.parts else None
            if status == "suspended":
                if not allow_suspended:
                    raise RuntimeError("trajectory suspended outside quiesce")
                suspended.append(sample)
            else:
                terminal.append(sample)
        self._buffer.extend(self._pending.add(terminal))
        return suspended

    def _apply_filter(self, samples: List["Sample"]) -> List["Sample"]:
        candidates = list(samples)
        kept = list(self._filter(candidates, self._weight_version))
        if kept and Counter(map(id, kept)) != Counter(map(id, candidates)):
            raise RuntimeError("agentic rollout filter must retain or discard an entire root")
        return kept

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RolloutManager is closed")


__all__ = ["RolloutManager", "RolloutUnderflow"]
