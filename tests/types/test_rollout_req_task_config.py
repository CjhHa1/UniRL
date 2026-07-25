"""Regression tests for the RolloutReq task-routing field rename."""

from __future__ import annotations

from dataclasses import fields

import pytest

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


def test_rollout_req_rejects_removed_stage_config_keyword() -> None:
    field_names = {item.name for item in fields(RolloutReq)}
    assert "task_config" in field_names
    assert "stage_config" not in field_names

    with pytest.raises(TypeError, match="stage_config"):
        RolloutReq(stage_config={"task": "it2i"})  # type: ignore[call-arg]
