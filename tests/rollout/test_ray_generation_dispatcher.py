from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import unirl.rollout.async_runtime as runtime
from unirl.distributed.group.dispatch import Dispatch
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack


class FakeTransport:
    localized: Any = None

    @classmethod
    def localize(
        cls,
        shards,
        pool,
        device_ids,
        worker_ids,
    ):
        cls.localized = (shards, pool, device_ids, worker_ids)
        return shards


class FakeHandle:
    def __init__(self) -> None:
        self.dp_size = 2
        self.pool = SimpleNamespace(transport_cls=FakeTransport)
        self.device_ids = [0, 1]
        self.worker_ids = ["id-0", "id-1"]
        self.workers = ["worker-0", "worker-1"]
        self.execute_call = None
        self.rebind_calls = []

    def _execute_all(
        self,
        method_name,
        shards,
        *,
        grad_mode,
        call_id,
    ):
        self.execute_call = (
            method_name,
            shards,
            grad_mode,
            call_id,
        )
        return ["ref-0", "ref-1"]

    def _rebind_tree(
        self,
        result,
        worker,
        *,
        worker_local,
    ):
        self.rebind_calls.append((result, worker, worker_local))
        return f"bound:{result}:{worker}"


def _request(batch_size: int) -> RolloutReq:
    sample_ids = [f"p{i}" for i in range(batch_size)]
    return RolloutReq(sample_ids=sample_ids, group_ids=sample_ids)


def test_dispatcher_mirrors_handle_dispatch_and_collect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = FakeHandle()
    dispatched = []
    collected = []

    def dispatch_fn(rollout, args, kwargs, batch_size):
        dispatched.append((rollout, args, kwargs, batch_size))
        return [(("shard-0",), {}), (("shard-1",), {})]

    expected = RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=["s0", "s1"],
                parent_ids=["p0", "p1"],
            )
        }
    )

    def collect_fn(rollout, results):
        collected.append((rollout, results))
        return expected

    monkeypatch.setitem(
        runtime.DISPATCH_MODE_REGISTRY,
        Dispatch.DP_SCATTER,
        {
            "dispatch_fn": dispatch_fn,
            "collect_fn": collect_fn,
        },
    )
    monkeypatch.setattr(
        runtime.ray,
        "get",
        lambda refs: ["output-0", "output-1"],
    )

    dispatcher = runtime.RayGenerationDispatcher(handle)
    req = _request(2)
    job = dispatcher.launch(req, gen_id=3, weight_version=4)
    resp = dispatcher.collect(job)

    assert dispatched == [(handle, (req,), {}, 2)]
    assert FakeTransport.localized == (
        [(("shard-0",), {}), (("shard-1",), {})],
        handle.pool,
        [0, 1],
        ["id-0", "id-1"],
    )
    assert handle.execute_call == (
        "generate",
        [(("shard-0",), {}), (("shard-1",), {})],
        False,
        None,
    )
    assert job.refs == ["ref-0", "ref-1"]
    assert job.req is req
    assert job.gen_id == 3
    assert job.weight_version == 4
    assert handle.rebind_calls == [
        ("output-0", "worker-0", False),
        ("output-1", "worker-1", False),
    ]
    assert collected == [
        (
            handle,
            [
                "bound:output-0:worker-0",
                "bound:output-1:worker-1",
            ],
        )
    ]
    assert resp is expected


def test_dispatcher_rejects_nondivisible_request() -> None:
    dispatcher = runtime.RayGenerationDispatcher(FakeHandle())

    with pytest.raises(
        ValueError,
        match="batch_size=1 not divisible by rollout dp_size=2",
    ):
        dispatcher.launch(_request(1), gen_id=0, weight_version=0)
