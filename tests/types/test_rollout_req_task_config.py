"""Regression tests for the RolloutReq task-routing field rename."""

from __future__ import annotations

from dataclasses import fields

import pytest
from omegaconf import OmegaConf

from unirl.train_diffusion import _resolve_task_config
from unirl.types.rollout_req import RolloutReq


def test_rollout_req_uses_task_config_as_shared_routing_metadata() -> None:
    task_config = {"task": "it2i", "ar": {"system_instruction": "edit the image"}}
    req = RolloutReq(
        sample_ids=["sample-0", "sample-1"],
        group_ids=["group-0", "group-0"],
        task_config=task_config,
    )

    assert req.task_config == task_config
    assert req.slice(0, 1).task_config == task_config


def test_rollout_req_keeps_deprecated_stage_config_as_nonserialized_alias() -> None:
    field_names = {item.name for item in fields(RolloutReq)}
    assert "task_config" in field_names
    assert "stage_config" not in field_names

    with pytest.warns(DeprecationWarning, match="task_config"):
        req = RolloutReq(stage_config={"task": "it2i"})
    assert req.task_config == {"task": "it2i"}
    assert req.stage_config is req.task_config
    assert req.slice(0, 0).stage_config == {"task": "it2i"}


def test_rollout_req_rejects_conflicting_new_and_deprecated_values() -> None:
    with pytest.raises(ValueError, match="conflicting task_config"):
        RolloutReq(
            task_config={"task": "t2i"},
            stage_config={"task": "it2i"},
        )


def test_train_entrypoint_accepts_deprecated_recipe_key_without_dropping_task() -> None:
    with pytest.warns(FutureWarning, match="task_config"):
        task_config = _resolve_task_config(OmegaConf.create({"stage_config": {"task": "it2i"}}))
    assert task_config == {"task": "it2i"}


def test_train_entrypoint_rejects_ambiguous_recipe_keys() -> None:
    with pytest.raises(ValueError, match="only task_config"):
        _resolve_task_config(
            OmegaConf.create(
                {
                    "task_config": {"task": "t2i"},
                    "stage_config": {"task": "it2i"},
                }
            )
        )
