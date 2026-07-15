from __future__ import annotations

from typing import List

import pytest

from unirl.rollout.async_runtime import (
    AsyncRolloutScheduler,
    InflightGeneration,
)
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack


def _request(gen_id: int) -> RolloutReq:
    sample_id = f"prompt-{gen_id}"
    return RolloutReq(sample_ids=[sample_id], group_ids=[sample_id])


def _response(gen_id: int) -> RolloutResp:
    group_id = f"group-{gen_id}"
    return RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=[f"{group_id}/s0"],
                parent_ids=[group_id],
            )
        }
    )


def _complete(
    job: InflightGeneration,
    resp: RolloutResp,
) -> List[RolloutResp]:
    del job
    return resp.split()


class FakeDispatcher:
    def __init__(self, *, ready_on_launch: bool = False) -> None:
        self.ready_on_launch = ready_on_launch
        self.ready: set[int] = set()
        self.launched: List[InflightGeneration] = []
        self.waited: List[int] = []
        self.collected: List[int] = []

    def launch(
        self,
        req: RolloutReq,
        *,
        gen_id: int,
        weight_version: int,
    ) -> InflightGeneration:
        job = InflightGeneration(
            refs=[gen_id],
            worker_local=False,
            req=req,
            gen_id=gen_id,
            weight_version=weight_version,
        )
        self.launched.append(job)
        if self.ready_on_launch:
            self.ready.add(gen_id)
        return job

    def is_ready(self, job: InflightGeneration) -> bool:
        return job.gen_id in self.ready

    def wait(self, job: InflightGeneration) -> None:
        self.waited.append(job.gen_id)
        self.ready.add(job.gen_id)

    def collect(self, job: InflightGeneration) -> RolloutResp:
        self.collected.append(job.gen_id)
        return _response(job.gen_id)


def test_ready_jobs_are_reaped_and_freshest_groups_selected() -> None:
    dispatcher = FakeDispatcher(ready_on_launch=True)
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_batch=2)

    picked = scheduler.next_batch(
        rollout_id=0,
        sync_interval=2,
        max_inflight=4,
        max_staleness=0,
        num_rollouts=4,
        current_version=7,
        build_req=_request,
        on_complete=_complete,
    )

    assert [item.gen_id for item in picked] == [1, 0]
    assert [job.weight_version for job in dispatcher.launched] == [7, 7]
    assert dispatcher.collected == [0, 1]
    assert scheduler.inflight_count == 0


@pytest.mark.parametrize(
    ("rollout_id", "sync_interval", "start_id", "expected_ceiling"),
    [
        (0, 1, 0, 1),
        (0, 2, 0, 2),
        (1, 2, 1, 2),
        (3, 2, 3, 4),
        (4, 3, 4, 6),
    ],
)
def test_on_policy_launch_clamp_stays_inside_current_sync_window(
    rollout_id: int,
    sync_interval: int,
    start_id: int,
    expected_ceiling: int,
) -> None:
    dispatcher = FakeDispatcher()
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_batch=1)
    scheduler.reset(start_id)

    scheduler.next_batch(
        rollout_id=rollout_id,
        sync_interval=sync_interval,
        max_inflight=16,
        max_staleness=0,
        num_rollouts=100,
        current_version=0,
        build_req=_request,
        on_complete=_complete,
    )

    launched_ids = [job.gen_id for job in dispatcher.launched]
    assert launched_ids
    assert max(launched_ids) < expected_ceiling
    assert scheduler.launch_id == expected_ceiling
    scheduler.drain_all(_complete)


def test_wait_blocks_on_oldest_job_then_reaps_it() -> None:
    dispatcher = FakeDispatcher()
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_batch=1)

    picked = scheduler.next_batch(
        rollout_id=0,
        sync_interval=1,
        max_inflight=1,
        max_staleness=0,
        num_rollouts=1,
        current_version=0,
        build_req=_request,
        on_complete=_complete,
    )

    assert [item.gen_id for item in picked] == [0]
    assert dispatcher.waited == [0]
    assert dispatcher.collected == [0]


def test_drain_all_quiesces_remaining_jobs() -> None:
    dispatcher = FakeDispatcher()
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_batch=1)

    scheduler.next_batch(
        rollout_id=0,
        sync_interval=2,
        max_inflight=2,
        max_staleness=0,
        num_rollouts=2,
        current_version=0,
        build_req=_request,
        on_complete=_complete,
    )
    assert scheduler.inflight_count == 1

    scheduler.drain_all(_complete)

    assert scheduler.inflight_count == 0
    assert scheduler.buffer_size == 1
    assert dispatcher.collected == [0, 1]


def test_drain_all_clears_inflight_state_when_callback_fails() -> None:
    dispatcher = FakeDispatcher()
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_batch=1)

    scheduler.next_batch(
        rollout_id=0,
        sync_interval=3,
        max_inflight=3,
        max_staleness=0,
        num_rollouts=3,
        current_version=0,
        build_req=_request,
        on_complete=_complete,
    )

    seen: List[int] = []

    def interrupt_first(
        job: InflightGeneration,
        resp: RolloutResp,
    ) -> List[RolloutResp]:
        seen.append(job.gen_id)
        if job.gen_id == 1:
            raise KeyboardInterrupt
        return resp.split()

    with pytest.raises(KeyboardInterrupt):
        scheduler.drain_all(interrupt_first)
    assert scheduler.inflight_count == 0
    assert seen == [1, 2]
    assert dispatcher.collected == [0, 1, 2]
    assert scheduler.buffer_size == 1


def test_reap_ready_processes_all_jobs_before_reraising_callback_error() -> None:
    dispatcher = FakeDispatcher(ready_on_launch=True)
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_batch=1)

    def fail_first(
        job: InflightGeneration,
        resp: RolloutResp,
    ) -> List[RolloutResp]:
        if job.gen_id == 0:
            raise RuntimeError("scoring failed")
        return resp.split()

    with pytest.raises(RuntimeError, match="scoring failed"):
        scheduler.next_batch(
            rollout_id=0,
            sync_interval=2,
            max_inflight=2,
            max_staleness=0,
            num_rollouts=2,
            current_version=0,
            build_req=_request,
            on_complete=fail_first,
        )

    assert dispatcher.collected == [0, 1]
    assert scheduler.inflight_count == 0
    assert scheduler.buffer_size == 1


def test_buffer_underflow_without_inflight_jobs_raises() -> None:
    dispatcher = FakeDispatcher()
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_batch=1)
    scheduler.reset(start_id=1)

    with pytest.raises(
        RuntimeError,
        match="buffer underflow with no in-flight",
    ):
        scheduler.next_batch(
            rollout_id=0,
            sync_interval=1,
            max_inflight=1,
            max_staleness=0,
            num_rollouts=1,
            current_version=0,
            build_req=_request,
            on_complete=_complete,
        )


def test_scheduler_rejects_invalid_group_batch_size() -> None:
    with pytest.raises(ValueError, match="groups_per_batch must be >= 1"):
        AsyncRolloutScheduler(FakeDispatcher(), groups_per_batch=0)
