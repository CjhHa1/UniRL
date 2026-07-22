"""EP-aware full-state checkpoint helpers for the VeOmni backend.

VeOmni pre-slices fused expert parameters before DTensor sharding: each EP
rank owns a DTensor whose *global* shape is ``[E / ep, ...]``. The EP axis is
therefore invisible to DCP's normal full-state gather/load. These helpers add
that missing outer dimension for single-file ``torch`` checkpoints:

* save: replicate each local ``[E / ep, ...]`` block over ``ep_fsdp``, gather
  the blocks over ``ep``, and replace DCP's rank-0 expert entry with
  ``[E, ...]``;
* load: slice the saved ``[E, ...]`` tensor for this EP rank, then let DCP
  shard that local block over ``ep_fsdp`` without a rank-0 broadcast.

The same transformation is applied to parameter-shaped optimizer state
(``exp_avg`` / ``exp_avg_sq``), so Adam resumes exactly with the model.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch import nn

from unirl.train.backend.sharded_state import (
    StateDict,
    gather_optimizer_state_dict,
    gather_state_dict,
    load_model_state_dict,
    load_optimizer_state_dict,
)

EP_CHECKPOINT_VERSION = 1


def has_ep_params(model: nn.Module) -> bool:
    """Whether ``model`` carries VeOmni's non-empty EP parameter group."""
    groups = getattr(model, "_extra_parallel_param_groups", None)
    return bool(groups and groups.get("ep"))


def ep_named_parameters(model: nn.Module) -> Tuple[Tuple[str, nn.Parameter], ...]:
    """Return EP parameters in deterministic model order.

    ``_extra_parallel_param_groups`` is attached by ``veomni_parallelize`` from
    the actual DTensor meshes, so it is the model-agnostic source of truth for
    Qwen3-MoE and HI3 alike.
    """
    groups = getattr(model, "_extra_parallel_param_groups", None) or {}
    ep_params = tuple(groups.get("ep", ()))
    ep_ids = {id(param) for param in ep_params}
    named = tuple((name, param) for name, param in model.named_parameters() if id(param) in ep_ids)
    if len(named) != len(ep_params):
        raise RuntimeError(
            "EP checkpoint: could not map every _extra_parallel_param_groups['ep'] "
            f"parameter to an FQN ({len(named)}/{len(ep_params)} mapped)."
        )
    return named


def gather_ep_model_state_dict(model: nn.Module) -> StateDict:
    """Gather a rank-0 full model state with complete stacked expert tensors."""
    full_state = gather_state_dict(model)
    ps = _parallel_state()
    rank0 = _current_rank() == 0

    for name, param in ep_named_parameters(model):
        local_block = _materialize_local_block(param)
        _validate_local_shape(name, local_block, param)
        stacked = _gather_stacked_block(local_block, ps)
        if rank0:
            if name not in full_state:
                raise RuntimeError(f"EP checkpoint: model state is missing expert parameter {name!r}.")
            full_state[name] = stacked.detach().cpu()

    return full_state


def load_ep_model_state_dict(
    model: nn.Module,
    state_dict: StateDict,
    *,
    strict: bool,
) -> None:
    """Slice full expert tensors for this EP rank and load without broadcasting."""
    ps = _parallel_state()
    local_state = dict(state_dict)
    for name, param in ep_named_parameters(model):
        if name not in state_dict:
            if strict:
                raise RuntimeError(f"EP checkpoint: checkpoint is missing expert parameter {name!r}.")
            continue
        local_state[name] = _slice_stacked_block(
            name,
            state_dict[name],
            local_shape=tuple(param.shape),
            ep_size=int(ps.ep_size),
            ep_rank=_ep_rank(ps),
        )

    # Every rank loaded the single checkpoint file and now carries its own
    # [E/ep,...] expert block. Broadcasting rank 0 here would recreate the bug.
    load_model_state_dict(
        model,
        local_state,
        strict=strict,
        broadcast_from_rank0=False,
    )


def gather_ep_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> StateDict:
    """Gather rank-0 optimizer state with full stacked EP moments."""
    full_state = gather_optimizer_state_dict(model, optimizer)
    ps = _parallel_state()
    rank0 = _current_rank() == 0
    rank0_entries = full_state.get("state", {}) if rank0 else {}

    for name, param in ep_named_parameters(model):
        local_entry = optimizer.state.get(param, {})
        for state_name in sorted(local_entry):
            value = local_entry[state_name]
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            if tuple(value.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"EP checkpoint: optimizer state {name!r}/{state_name!r} has "
                    f"shape {tuple(value.shape)}, expected parameter shape {tuple(param.shape)}."
                )
            local_block = _materialize_local_block(value)
            _validate_local_shape(f"{name}/{state_name}", local_block, param)
            stacked = _gather_stacked_block(local_block, ps)
            if rank0:
                entry = rank0_entries.get(name)
                if entry is None or state_name not in entry:
                    raise RuntimeError(f"EP checkpoint: gathered optimizer state is missing {name!r}/{state_name!r}.")
                entry[state_name] = stacked.detach().cpu()

    return full_state


def load_ep_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state_dict: StateDict,
) -> None:
    """Slice full EP optimizer moments and reshard without rank-0 broadcast."""
    ps = _parallel_state()
    local_state = dict(state_dict)
    source_entries = state_dict.get("state", {})
    local_entries: Dict[str, object] = dict(source_entries)

    for name, param in ep_named_parameters(model):
        source_entry = source_entries.get(name)
        if source_entry is None:
            # Adam has no entry before its first step; that is a valid empty state.
            continue
        local_entry = dict(source_entry)
        for state_name, value in source_entry.items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            local_entry[state_name] = _slice_stacked_block(
                f"{name}/{state_name}",
                value,
                local_shape=tuple(param.shape),
                ep_size=int(ps.ep_size),
                ep_rank=_ep_rank(ps),
            )
        local_entries[name] = local_entry

    local_state["state"] = local_entries
    load_optimizer_state_dict(
        model,
        optimizer,
        local_state,
        broadcast_from_rank0=False,
    )


def _parallel_state():
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    if not getattr(ps, "ep_enabled", False) or int(ps.ep_size) <= 1:
        raise RuntimeError("EP checkpoint helper called while expert parallelism is disabled.")
    return ps


def _ep_rank(ps) -> int:
    if hasattr(ps, "extra_parallel_rank"):
        return int(ps.extra_parallel_rank("ep"))
    return int(ps.ep_rank)


def _materialize_local_block(value: torch.Tensor) -> torch.Tensor:
    """Replicate a DTensor over its own mesh, yielding this EP rank's block."""
    try:
        from torch.distributed.tensor import DTensor, Replicate
    except ImportError:  # pragma: no cover - older torch
        from torch.distributed._tensor import DTensor, Replicate

    if isinstance(value, DTensor):
        work = value
        if work.to_local().device.type == "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError("EP checkpoint: cannot gather a CPU DTensor over an NCCL mesh without CUDA.")
            work = work.cuda()
        placements = [Replicate()] * work.device_mesh.ndim
        return work.redistribute(placements=placements).to_local().detach()
    return value.detach()


def _gather_stacked_block(local_block: torch.Tensor, ps) -> torch.Tensor:
    """Gather contiguous expert blocks in EP-rank order and concatenate dim 0."""
    block = local_block.contiguous()
    if block.device.type == "cpu" and dist.is_initialized() and dist.get_backend(ps.ep_group) == "nccl":
        block = block.cuda()
    gathered = [torch.empty_like(block) for _ in range(int(ps.ep_size))]
    dist.all_gather(gathered, block, group=ps.ep_group)
    return torch.cat(gathered, dim=0)


def _slice_stacked_block(
    name: str,
    value: object,
    *,
    local_shape: Tuple[int, ...],
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"EP checkpoint: {name!r} must be a tensor, got {type(value).__name__}.")
    expected = (local_shape[0] * ep_size, *local_shape[1:])
    actual = tuple(value.shape)
    if actual != expected:
        if actual == local_shape:
            raise RuntimeError(
                f"EP checkpoint: {name!r} contains only one EP block {actual}; "
                "this checkpoint was written by the old corrupt full-state path "
                "and cannot recover the missing expert ranks."
            )
        raise RuntimeError(f"EP checkpoint: {name!r} has shape {actual}, expected full stacked shape {expected}.")
    start = ep_rank * local_shape[0]
    block = value[start : start + local_shape[0]].contiguous()
    if tuple(block.shape) != local_shape:
        raise RuntimeError(f"EP checkpoint: sliced {name!r} to {tuple(block.shape)}, expected {local_shape}.")
    return block


def _validate_local_shape(name: str, block: torch.Tensor, param: nn.Parameter) -> None:
    if tuple(block.shape) != tuple(param.shape):
        raise RuntimeError(
            f"EP checkpoint: materialized local block {name!r} has shape "
            f"{tuple(block.shape)}, expected {tuple(param.shape)}."
        )


def _current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = [
    "EP_CHECKPOINT_VERSION",
    "has_ep_params",
    "ep_named_parameters",
    "gather_ep_model_state_dict",
    "load_ep_model_state_dict",
    "gather_ep_optimizer_state_dict",
    "load_ep_optimizer_state_dict",
]
