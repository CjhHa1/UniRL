from __future__ import annotations

from dataclasses import dataclass

import pytest

from unirl.rollout.engine.asynchronous import AsyncBatchRolloutEngine


@dataclass
class _Pending:
    value: str

    def ready(self) -> bool:
        return True

    def result(self) -> str:
        return self.value

    def wait(self) -> None:
        return None


class _RolloutHandle:
    def __init__(self) -> None:
        self.launched: list[str] = []

    def launch_nowait(self, method: str, sample: str) -> _Pending:
        assert method == "generate"
        self.launched.append(sample)
        return _Pending(sample)


def test_batch_engine_stamps_train_version_and_consumes_generation_fifo() -> None:
    rollout = _RolloutHandle()
    engine = AsyncBatchRolloutEngine(
        rollout,
        process_completion=lambda _gen_id, completed: [f"{completed}/0", f"{completed}/1"],
        groups_per_batch=2,
    )

    engine.submit("batch-0", behavior_version=0)
    engine.submit("batch-1", behavior_version=4)
    assert engine.poll() == 2

    first = engine.pop_next_batch(train_version=4, max_policy_lag=4)
    second = engine.pop_next_batch(train_version=4, max_policy_lag=4)
    assert first is not None and first.groups == ["batch-0/0", "batch-0/1"]
    assert first.behavior_version == 0
    assert second is not None and second.groups == ["batch-1/0", "batch-1/1"]
    assert second.behavior_version == 4


def test_batch_engine_rejects_non_atomic_generation() -> None:
    engine = AsyncBatchRolloutEngine(
        _RolloutHandle(),
        process_completion=lambda _gen_id, completed: [completed],
        groups_per_batch=2,
    )
    engine.submit("short", behavior_version=0)
    with pytest.raises(RuntimeError, match="expected groups_per_batch=2"):
        engine.poll()


def test_batch_engine_fails_closed_on_stale_ready_batch() -> None:
    engine = AsyncBatchRolloutEngine(
        _RolloutHandle(),
        process_completion=lambda _gen_id, completed: [completed],
        groups_per_batch=1,
    )
    engine.submit("old", behavior_version=0)
    engine.poll()
    with pytest.raises(RuntimeError, match="exceeded policy lag budget"):
        engine.pop_next_batch(train_version=2, max_policy_lag=1)
