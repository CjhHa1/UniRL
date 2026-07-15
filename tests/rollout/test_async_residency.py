from __future__ import annotations

import time
from types import SimpleNamespace
from typing import List

import pytest

from unirl.rollout.async_runtime import AsyncRolloutScheduler, InflightGeneration
from unirl.trainer.async_ar import AsyncARTrainer
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack


def _request(gen_id: int) -> RolloutReq:
    sample_id = f"prompt-{gen_id}"
    return RolloutReq(sample_ids=[sample_id], group_ids=[sample_id])


def _response(gen_id: int, groups: int) -> RolloutResp:
    group_ids = [f"gen-{gen_id}/group-{index}" for index in range(groups)]
    return RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=[f"{group_id}/s0" for group_id in group_ids],
                parent_ids=group_ids,
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
    def __init__(
        self,
        *,
        groups_per_response: int,
        ready_on_launch: bool = False,
        wait_delay_s: float = 0.0,
    ) -> None:
        self.groups_per_response = groups_per_response
        self.ready_on_launch = ready_on_launch
        self.wait_delay_s = float(wait_delay_s)
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
        if self.wait_delay_s:
            time.sleep(self.wait_delay_s)
        self.waited.append(job.gen_id)
        self.ready.add(job.gen_id)

    def collect(self, job: InflightGeneration) -> RolloutResp:
        self.collected.append(job.gen_id)
        return _response(job.gen_id, self.groups_per_response)


def _builder(calls: List[int]):
    def build(gen_id: int) -> RolloutReq:
        calls.append(gen_id)
        return _request(gen_id)

    return build


def test_prefetch_zero_admits_only_the_current_batch() -> None:
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=4)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=4,
        groups_per_generation=4,
        prefetch_batches=0,
    )

    scheduler.next_batch(
        rollout_id=0,
        sync_interval=2,
        max_inflight=4,
        max_staleness=0,
        num_rollouts=10,
        current_version=0,
        build_req=_builder(calls),
        on_complete=_complete,
    )

    assert calls == [0]
    assert scheduler.inflight_count == 0
    assert scheduler.buffer_size == 0


def test_prefetch_one_launches_one_successor_with_single_inflight_slot() -> None:
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=4)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=4,
        groups_per_generation=4,
        prefetch_batches=1,
    )

    scheduler.next_batch(
        rollout_id=0,
        sync_interval=2,
        max_inflight=1,
        max_staleness=0,
        num_rollouts=10,
        current_version=0,
        build_req=_builder(calls),
        on_complete=_complete,
    )

    assert calls == [0, 1]
    assert scheduler.inflight_count == 1
    metrics = scheduler.drain_metrics(current_version=0)
    assert metrics["async/reserved_groups"] == 4
    scheduler.drain_all(_complete)


def test_sync_window_clamp_can_block_prefetch() -> None:
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=4)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=4,
        groups_per_generation=4,
        prefetch_batches=1,
    )

    scheduler.next_batch(
        rollout_id=0,
        sync_interval=1,
        max_inflight=1,
        max_staleness=0,
        num_rollouts=10,
        current_version=0,
        build_req=_builder(calls),
        on_complete=_complete,
    )

    assert calls == [0]
    assert scheduler.inflight_count == 0


def test_capacity_limits_admission_independently_from_max_inflight() -> None:
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=4)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=4,
        groups_per_generation=4,
        prefetch_batches=1,
    )

    scheduler.next_batch(
        rollout_id=0,
        sync_interval=10,
        max_inflight=8,
        max_staleness=0,
        num_rollouts=10,
        current_version=0,
        build_req=_builder(calls),
        on_complete=_complete,
    )

    assert calls == [0, 1]
    assert max(job.gen_id for job in dispatcher.launched) == 1
    scheduler.drain_all(_complete)


def test_bounded_prefetch_validates_fixed_generation_yield() -> None:
    dispatcher = FakeDispatcher(groups_per_response=1)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=2,
        groups_per_generation=2,
        prefetch_batches=0,
    )

    with pytest.raises(RuntimeError, match="produced 1 root group"):
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


@pytest.mark.parametrize(
    ("groups_per_batch", "groups_per_generation", "message"),
    [
        (4, None, "requires groups_per_generation"),
        (4, 1, "produce exactly one training batch"),
        (4, 5, "produce exactly one training batch"),
    ],
)
def test_bounded_prefetch_rejects_invalid_group_units(
    groups_per_batch: int,
    groups_per_generation: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AsyncRolloutScheduler(
            FakeDispatcher(groups_per_response=1),
            groups_per_batch=groups_per_batch,
            groups_per_generation=groups_per_generation,
            prefetch_batches=1,
        )


def test_stale_eviction_is_observable_and_does_not_block_admission() -> None:
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=1, ready_on_launch=True)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=1,
        groups_per_generation=1,
        prefetch_batches=1,
    )

    scheduler.next_batch(
        rollout_id=0,
        sync_interval=10,
        max_inflight=2,
        max_staleness=0,
        num_rollouts=4,
        current_version=0,
        build_req=_builder(calls),
        on_complete=_complete,
    )
    first_metrics = scheduler.drain_metrics(current_version=0)
    assert first_metrics["async/resident_groups"] == 1

    scheduler.next_batch(
        rollout_id=1,
        sync_interval=10,
        max_inflight=2,
        max_staleness=0,
        num_rollouts=4,
        current_version=1,
        build_req=_builder(calls),
        on_complete=_complete,
    )
    metrics = scheduler.drain_metrics(current_version=1)

    assert metrics["async/evicted_stale_groups"] == 1
    assert metrics["async/admitted_jobs"] > 0
    assert metrics["async/dropped_groups"] == 0


def test_hard_launch_ceiling_drains_state_at_checkpoint_boundary() -> None:
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=1, ready_on_launch=True)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=1,
        groups_per_generation=1,
        prefetch_batches=1,
    )

    scheduler.next_batch(
        rollout_id=0,
        sync_interval=10,
        max_inflight=2,
        max_staleness=0,
        num_rollouts=10,
        current_version=0,
        build_req=_builder(calls),
        on_complete=_complete,
        hard_launch_ceiling=2,
    )
    scheduler.next_batch(
        rollout_id=1,
        sync_interval=10,
        max_inflight=2,
        max_staleness=0,
        num_rollouts=10,
        current_version=0,
        build_req=_builder(calls),
        on_complete=_complete,
        hard_launch_ceiling=2,
    )
    scheduler.drain_all(_complete)

    assert calls == [0, 1]
    assert scheduler.inflight_count == 0
    assert scheduler.buffer_size == 0


def test_hard_launch_ceiling_also_protects_legacy_admission() -> None:
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=1, ready_on_launch=True)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=1,
        groups_per_generation=1,
        prefetch_batches=None,
    )

    for rollout_id in range(2):
        scheduler.next_batch(
            rollout_id=rollout_id,
            sync_interval=10,
            max_inflight=4,
            max_staleness=0,
            num_rollouts=10,
            current_version=0,
            build_req=_builder(calls),
            on_complete=_complete,
            hard_launch_ceiling=2,
        )
    scheduler.drain_all(_complete)

    assert calls == [0, 1]
    assert scheduler.inflight_count == 0
    assert scheduler.buffer_size == 0


def test_none_prefetch_preserves_legacy_admission_sequence() -> None:
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=1, ready_on_launch=True)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=1,
        groups_per_generation=1,
        prefetch_batches=None,
    )

    picked = scheduler.next_batch(
        rollout_id=0,
        sync_interval=2,
        max_inflight=2,
        max_staleness=0,
        num_rollouts=10,
        current_version=0,
        build_req=_builder(calls),
        on_complete=_complete,
    )

    assert calls == [0, 1]
    assert [item.gen_id for item in picked] == [1]
    assert scheduler.buffer_size == 1


def test_quiesce_wait_and_reap_are_reported_in_same_metrics_drain() -> None:
    dispatcher = FakeDispatcher(
        groups_per_response=1,
        wait_delay_s=0.001,
    )
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=1,
        groups_per_generation=1,
        prefetch_batches=1,
    )
    scheduler.next_batch(
        rollout_id=0,
        sync_interval=2,
        max_inflight=1,
        max_staleness=0,
        num_rollouts=2,
        current_version=0,
        build_req=_request,
        on_complete=_complete,
    )
    scheduler.drain_metrics(current_version=0)

    scheduler.drain_all(_complete)
    metrics = scheduler.drain_metrics(current_version=0)

    assert metrics["async/quiesce_wait_time_s"] >= 0.001
    assert metrics["async/reaped_jobs"] == 1
    assert metrics["async/buffered_groups"] == 1


@pytest.mark.parametrize(
    ("prefetch_batches", "max_inflight", "sync_interval", "max_staleness"),
    [
        (0, 4, 1, 0),
        (1, 1, 2, 0),
        (1, 2, 2, 0),
        (2, 4, 3, 0),
        (2, 4, 2, 1),
    ],
)
def test_bounded_runtime_completes_multi_step_schedule_without_leaks(
    prefetch_batches: int,
    max_inflight: int,
    sync_interval: int,
    max_staleness: int,
) -> None:
    num_rollouts = 8
    calls: List[int] = []
    dispatcher = FakeDispatcher(groups_per_response=1, ready_on_launch=True)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=1,
        groups_per_generation=1,
        prefetch_batches=prefetch_batches,
    )
    version = 0

    for rollout_id in range(num_rollouts):
        scheduler.next_batch(
            rollout_id=rollout_id,
            sync_interval=sync_interval,
            max_inflight=max_inflight,
            max_staleness=max_staleness,
            num_rollouts=num_rollouts,
            current_version=version,
            build_req=_builder(calls),
            on_complete=_complete,
        )
        if (rollout_id + 1) % sync_interval == 0:
            scheduler.drain_all(_complete)
            version += 1
    scheduler.drain_all(_complete)

    assert calls == list(range(num_rollouts))
    assert scheduler.inflight_count == 0
    assert scheduler.buffer_size == 0


def test_trainer_checkpoint_helpers_cover_legacy_and_require_sync_alignment() -> None:
    trainer = object.__new__(AsyncARTrainer)
    trainer.weight_sync = object()

    assert (
        trainer._checkpoint_launch_ceiling(
            rollout_id=3,
            save_interval=5,
            num_rollouts=12,
        )
        == 5
    )
    trainer._validate_checkpoint_alignment(
        num_rollouts=12,
        save_interval=4,
        weight_sync_interval=2,
    )
    trainer._validate_checkpoint_alignment(
        num_rollouts=12,
        save_interval=100,
        weight_sync_interval=2,
    )
    with pytest.raises(ValueError, match="checkpoint steps must align"):
        trainer._validate_checkpoint_alignment(
            num_rollouts=12,
            save_interval=3,
            weight_sync_interval=2,
        )
    with pytest.raises(ValueError, match="checkpoint steps must align"):
        trainer._validate_checkpoint_alignment(
            num_rollouts=11,
            save_interval=4,
            weight_sync_interval=2,
        )

    trainer._async_scheduler = SimpleNamespace(inflight_count=0, buffer_size=1)
    with pytest.raises(RuntimeError, match="runtime is not empty"):
        trainer._assert_checkpoint_runtime_empty()


def test_metrics_drain_resets_cumulative_counters_only() -> None:
    dispatcher = FakeDispatcher(groups_per_response=1, ready_on_launch=True)
    scheduler = AsyncRolloutScheduler(
        dispatcher,
        groups_per_batch=1,
        groups_per_generation=1,
        prefetch_batches=1,
    )
    scheduler.next_batch(
        rollout_id=0,
        sync_interval=2,
        max_inflight=2,
        max_staleness=0,
        num_rollouts=2,
        current_version=3,
        build_req=_request,
        on_complete=_complete,
    )

    first = scheduler.drain_metrics(current_version=3)
    second = scheduler.drain_metrics(current_version=3)

    assert first["async/admitted_jobs"] == 2
    assert first["async/reaped_jobs"] == 2
    assert first["async/selected_groups"] == 1
    assert second["async/admitted_jobs"] == 0
    assert second["async/reaped_jobs"] == 0
    assert second["async/selected_groups"] == 0
    assert second["async/resident_groups"] == first["async/resident_groups"]
