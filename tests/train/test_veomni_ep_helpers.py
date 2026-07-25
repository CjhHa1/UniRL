import pytest
import torch
from torch import nn

from unirl.models.types.meta_init import restore_init_state
from unirl.train.backend.sharded_load import _read_ep_checkpoint_block
from unirl.train.backend.veomni.backend import _validate_ep_size
from unirl.train.backend.veomni.ep.checkpoint import _slice_stacked_block
from unirl.train.backend.veomni.ep.models.qwen3_moe import (
    build_local_fused_block,
    iter_hf_expert_tensors,
)


@pytest.mark.parametrize(("value", "expected"), [(1, 1), ("2", 2), (4, 4)])
def test_validate_ep_size_accepts_positive_divisors(value, expected):
    assert _validate_ep_size(value, world_size=8) == expected


@pytest.mark.parametrize("value", [0, -1, None, "bad", 3])
def test_validate_ep_size_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _validate_ep_size(value, world_size=8)


def test_restore_init_state_rejects_shape_drift():
    model = nn.Module()
    model.register_buffer("table", torch.zeros(2), persistent=False)

    with pytest.raises(RuntimeError, match="does not match live local shape"):
        restore_init_state(
            model,
            {"buffers": {"table": torch.zeros(3)}, "attrs": {}},
        )


def test_plan_declared_ep_slice_and_resume_slice_match():
    full = torch.arange(24).reshape(6, 4)

    class Slice:
        def __getitem__(self, index):
            return full[index]

    loaded = _read_ep_checkpoint_block(
        Slice(),
        name="experts",
        ckpt_shape=(6, 4),
        global_shape=(2, 4),
        ep_size=3,
        ep_rank=1,
        ep_dim=0,
    )
    resumed = _slice_stacked_block(
        "experts",
        full,
        local_shape=(2, 4),
        ep_size=3,
        ep_rank=1,
    )
    torch.testing.assert_close(loaded, full[2:4])
    torch.testing.assert_close(resumed, loaded)


def test_plan_declared_ep_slice_supports_nonzero_axis():
    full = torch.arange(24).reshape(2, 12)

    class Slice:
        def __getitem__(self, index):
            return full[index]

    loaded = _read_ep_checkpoint_block(
        Slice(),
        name="experts",
        ckpt_shape=(2, 12),
        global_shape=(2, 4),
        ep_size=3,
        ep_rank=2,
        ep_dim=1,
    )
    torch.testing.assert_close(loaded, full[:, 8:12])


def test_qwen3_moe_hf_fused_round_trip():
    prefix = "model.layers.0.mlp"
    source = {}
    for expert in range(4):
        source[f"{prefix}.experts.{expert}.gate_proj.weight"] = torch.full((3, 2), expert)
        source[f"{prefix}.experts.{expert}.up_proj.weight"] = torch.full((3, 2), expert + 10)
        source[f"{prefix}.experts.{expert}.down_proj.weight"] = torch.full((2, 3), expert + 20)

    gate_up = []
    down = []
    for ep_rank in range(2):
        gate_up.append(
            build_local_fused_block(
                fused_param_name=f"{prefix}.experts.gate_up_proj",
                expected_shape=(2, 6, 2),
                ep_rank=ep_rank,
                available_keys=set(source),
                get_tensor=source.__getitem__,
            )
        )
        down.append(
            build_local_fused_block(
                fused_param_name=f"{prefix}.experts.down_proj",
                expected_shape=(2, 2, 3),
                ep_rank=ep_rank,
                available_keys=set(source),
                get_tensor=source.__getitem__,
            )
        )

    recovered = dict(iter_hf_expert_tensors(f"{prefix}.experts.gate_up_proj", torch.cat(gate_up)))
    recovered.update(dict(iter_hf_expert_tensors(f"{prefix}.experts.down_proj", torch.cat(down))))
    assert recovered.keys() == source.keys()
    for name, tensor in source.items():
        torch.testing.assert_close(recovered[name], tensor)
