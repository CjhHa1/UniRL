"""Small EP checkpoint round-trip using synthetic expert tensors.

This exercises the production single-file model + Adam-state helpers without a
large MoE checkpoint or fused kernels. It defaults to CPU/Gloo; set
``DEVICE=cuda`` for NCCL:

    torchrun --nproc_per_node=4 scripts/ep_verify/unirl_ep_checkpoint_roundtrip.py \
        2 /tmp/unirl_ep_checkpoint_roundtrip.pt
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch import nn

from unirl.train.backend.veomni.ep.checkpoint import (
    gather_ep_model_state_dict,
    gather_ep_optimizer_state_dict,
    load_ep_model_state_dict,
    load_ep_optimizer_state_dict,
)


class _TinyExpertModel(nn.Module):
    def __init__(self, *, ep_rank: int, device: torch.device) -> None:
        super().__init__()
        # This is deliberately the VeOmni representation: dim 0 is already the
        # local E/ep expert block, outside any DTensor placement.
        self.experts = nn.Parameter(torch.full((2, 3), float(ep_rank + 1), device=device))
        self._extra_parallel_param_groups = {
            "ep": [self.experts],
            "non_extra_parallel": [],
        }


def main() -> None:
    ep_size = int(sys.argv[1])
    checkpoint_path = sys.argv[2]
    local_rank = int(os.environ["LOCAL_RANK"])
    device_type = os.environ.get("DEVICE", "cpu").strip().lower()
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif device_type == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError(f"DEVICE must be 'cpu' or 'cuda', got {device_type!r}")
    dist.init_process_group(backend)

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state

    world = dist.get_world_size()
    if world % ep_size:
        raise ValueError(f"world_size={world} must be divisible by ep_size={ep_size}")
    init_parallel_state(
        dp_size=world,
        ulysses_size=1,
        dp_mode="fsdp2",
        device_type=device_type,
        extra_parallel_sizes=(ep_size,),
        extra_parallel_names=("ep",),
    )
    ps = get_parallel_state()
    ep_rank = int(ps.extra_parallel_rank("ep"))

    model = _TinyExpertModel(ep_rank=ep_rank, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    model.experts.grad = torch.full_like(model.experts, 0.1 * (ep_rank + 1))
    optimizer.step()

    expected_param = model.experts.detach().clone()
    expected_optim = {
        key: value.detach().clone()
        for key, value in optimizer.state[model.experts].items()
        if isinstance(value, torch.Tensor)
    }

    model_state = gather_ep_model_state_dict(model)
    optimizer_state = gather_ep_optimizer_state_dict(model, optimizer)
    if dist.get_rank() == 0:
        expected_shape = (expected_param.shape[0] * ep_size, *expected_param.shape[1:])
        assert tuple(model_state["experts"].shape) == expected_shape
        assert tuple(optimizer_state["state"]["experts"]["exp_avg"].shape) == expected_shape
        torch.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_path)
    dist.barrier()

    with torch.no_grad():
        model.experts.zero_()
        for value in optimizer.state[model.experts].values():
            if isinstance(value, torch.Tensor):
                value.zero_()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_ep_model_state_dict(model, checkpoint["model"], strict=True)
    load_ep_optimizer_state_dict(model, optimizer, checkpoint["optimizer"])

    torch.testing.assert_close(model.experts, expected_param, rtol=0, atol=0)
    for key, expected in expected_optim.items():
        torch.testing.assert_close(optimizer.state[model.experts][key], expected, rtol=0, atol=0)

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"EP checkpoint round-trip PASS (world={world}, ep={ep_size})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
