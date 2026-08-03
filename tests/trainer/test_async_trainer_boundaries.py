from __future__ import annotations

from unirl.trainer.async_ar import AsyncARTrainer
from unirl.trainer.async_policy import PolicyVersionState


class _Engine:
    def __init__(self) -> None:
        self.ready_count = 0
        self.quiesces = 0

    def quiesce(self) -> None:
        self.quiesces += 1


class _WeightSync:
    def __init__(self) -> None:
        self.calls = 0

    def sync(self) -> None:
        self.calls += 1


class _Rollout:
    def __init__(self) -> None:
        self.versions: list[int] = []

    def set_policy_version(self, train_version: int) -> None:
        self.versions.append(train_version)


def test_async_ar_baseline_eval_syncs_once_even_when_version_is_zero() -> None:
    trainer = object.__new__(AsyncARTrainer)
    trainer._policy_versions = PolicyVersionState()
    trainer._rollout_initialized = False
    trainer._async_engine = _Engine()
    trainer.weight_sync = _WeightSync()
    trainer.rollout = _Rollout()

    trainer._prepare_rollout(sync_weights=True)
    trainer._prepare_rollout(sync_weights=True)

    assert trainer.weight_sync.calls == 1
    assert trainer.rollout.versions == [0]


def test_async_ar_sync_assigns_exact_train_version() -> None:
    trainer = object.__new__(AsyncARTrainer)
    trainer._policy_versions = PolicyVersionState(train_version=7, rollout_version=0)
    trainer._rollout_initialized = False
    trainer._async_engine = _Engine()
    trainer.weight_sync = _WeightSync()
    trainer.rollout = _Rollout()

    assert trainer._sync_rollout_weights() is True
    assert trainer._policy_versions.rollout_version == 7
    assert trainer.rollout.versions == [7]
