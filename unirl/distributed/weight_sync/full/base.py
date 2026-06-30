"""Base class for v2 full-(base-)weight sync handlers.

Counterpart to ``weight_sync/lora.py`` (LoRA-only). A ``FullWeightSync``
lives on the TRAIN slab as a sibling ``Remote`` of the FSDP backend (and, in
shared-process colocate, of the rollout engine) and pushes the freshly-trained
*full* weights into the rollout engine(s).

Subclasses pick a transport:
  - ``nccl.NCCLWeightSync``   — separate slabs (cross-node capable).
  - ``tensor.TensorWeightSync`` — colocate (serialized-tensor handoff).
  - ``ipc.IPCWeightSync``     — colocate (bucketed CUDA-IPC over ZMQ).

The base provides the transport-agnostic weight walk: full-tensor
materialization (FSDP shard → replicated, a train-mesh collective) and
size-bounded bucketing. All model / torch-heavy imports are deferred into the
methods so the driver can import this module to reference the class for
``remote(...)`` without eagerly pulling torch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from unirl.config.require import require
from unirl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


def _one_star(s: object) -> bool:
    """True if ``s`` is a string containing exactly one ``"*"``."""
    return isinstance(s, str) and s.count("*") == 1


def _validate_name_remap(
    name_remap: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    """Validate + return the ordered ``name_remap`` rewrite rules (fail-closed)."""
    rules = dict(name_remap or {})
    keys = list(rules.keys())
    for i, (key, value) in enumerate(rules.items()):
        require(_one_star(key), f"name_remap key {key!r} needs one '*'.")
        require(value is None or _one_star(value), f"name_remap[{key!r}] value: null or one '*'; got {value!r}.")
        require(key != "*" or i == len(keys) - 1, f"name_remap '*' must be last; shadows {keys[i + 1 :]!r}.")
    return rules


def _apply_name_remap(name: str, name_remap: Dict[str, Optional[str]]) -> Optional[str]:
    """Apply the ordered single-``*`` rewrite; return the new name or None to drop."""
    for key, value in name_remap.items():
        pre, _, post = key.partition("*")
        # length guard: pre and post must not overlap inside name
        if name.startswith(pre) and name.endswith(post) and len(name) >= len(pre) + len(post):
            if value is None:
                return None
            vpre, _, vpost = value.partition("*")
            return vpre + name[len(pre) : len(name) - len(post)] + vpost
    return name


class FullWeightSync(Remote):
    """Base for full-weight sync Remotes.

    Subclasses implement ``sync()`` (push current weights) plus whatever
    one-time connection setup their transport needs. ``backend`` is the FSDP
    backend sibling whose ``.model`` is the weight source.

    ``lora_merged`` selects what gets pushed: ``False`` (default) syncs the raw
    base weights (meaningful for full fine-tuning); ``True`` folds the trained
    LoRA deltas into the base weights and pushes the merged full model
    (meaningful for a LoRA run served without a separate adapter).

    ``adapter_name`` names which PEFT adapter's delta is folded/pushed. ``None``
    (default) defers to the backend's ``rollout_adapter_name`` — the EMA shadow
    (``"old"``) for DiffusionNFT, which is what carries the EMA-smoothed weights
    into a SEPARATE rollout engine where the in-process ``apply_eval_ema`` swap
    cannot reach; ``"default"`` otherwise. A non-default adapter REQUIRES
    ``lora_merged=True``: a full-weight receiver loads merged base weights and
    does not apply a separately-shipped delta, so the shadow must be folded in
    before the push (the ctor fails closed otherwise).

    ``track_prefix`` routes the push to one child of a ``ComposedRolloutEngine``
    (e.g. ``"ar"`` / ``"diffusion"``); empty (default) targets a single-model
    engine. Each transport forwards it to the rollout so the receiver demuxes.

    ``wire_dtype`` (e.g. ``"bf16"``) casts floating tensors to the rollout
    engine's dtype in the weight walk — shard-side, BEFORE the FSDP
    all-gather — so the gather, the bucket sizing, and every transport move
    wire-width bytes. Set it in the ``sync:`` block whenever the training
    masters are wider than the engine (``model_precision: fp32`` →
    ``wire_dtype: bf16``). Receivers ``copy_``-cast on load, so this is a
    bandwidth/memory policy, not a correctness one; ``None`` (default) ships
    tensors as-is.
    """

    def __init__(
        self,
        *,
        backend,
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        super().__init__()
        # Deferred: unirl.utils.dtypes imports torch at module scope, and this
        # module must stay driver-importable without torch.
        from unirl.utils.dtypes import parse_torch_dtype

        self._backend = backend
        self._bucket_bytes = int(bucket_size_mb) * 1024 * 1024
        self._flush_cache = bool(flush_cache)
        self._lora_merged = bool(lora_merged)
        # Which PEFT adapter's weights to push. ``None`` defers to the backend's
        # single source of truth (``rollout_adapter_name``): the EMA shadow
        # ("old") for DiffusionNFT, else "default". See class docstring; a
        # non-default adapter must be merged in, so fail closed.
        self._adapter_name = str(adapter_name) if adapter_name is not None else str(backend.rollout_adapter_name)
        if self._adapter_name != "default" and not self._lora_merged:
            raise ValueError(
                f"FullWeightSync: adapter_name={self._adapter_name!r} requires "
                f"lora_merged=True (a non-default adapter must be folded into the "
                f"pushed base weights; got lora_merged=False)."
            )
        # The pushed adapter is invisible in the recipe when it defaults from the
        # backend (DiffusionNFT resolves to the EMA shadow "old"), so log it once:
        # a SEPARATE rollout engine sampling under "old" is the intended faithful
        # path, not a misconfiguration.
        if self._adapter_name != "default":
            logger.info(
                "%s: rollout engine will be synced from adapter %r (%s).",
                type(self).__name__,
                self._adapter_name,
                "explicit adapter_name" if adapter_name is not None else "backend.rollout_adapter_name",
            )
        # Ordered, first-match-wins name rewrites (glob -> replacement, or None to
        # drop); see _validate_name_remap / _apply_name_remap for the contract.
        self._name_remap = _validate_name_remap(name_remap)
        # Routes the update to one child of a ComposedRolloutEngine; empty for
        # a single-model engine. Forwarded by each transport's ``sync()``.
        self._track_prefix = str(track_prefix or "")
        # Wire dtype for the weight walk (None = ship as-is); see class docstring.
        self._wire_dtype = parse_torch_dtype(wire_dtype, field_name="wire_dtype", allow_none=True)
        self.weight_version = 0

    # ------------------------------------------------------------------
    # Transport-agnostic weight walk
    # ------------------------------------------------------------------

    def _iter_full_tensors(self) -> Iterator[Tuple[str, "object"]]:
        """Yield ``(name, full_tensor)`` one at a time (lazy → bounded memory).

        Both walks apply ``_to_full_tensor`` (redistribute each FSDP ``DTensor``
        shard to Replicate → a collective over the train mesh), so this MUST run
        on every train rank in lockstep; each yields a full (unsharded) CUDA
        tensor per param, in a deterministic order.

        ``lora_merged`` selects the walk:
          - ``True``  → ``merged_state_dict`` folds LoRA deltas into the base
            weights and yields the trained module's own keys (LoRA already
            absorbed, ``.base_layer.`` flattened away).
          - ``False`` → ``raw_state_dict`` base weights, skipping
            ``.lora_A``/``.lora_B``.

        ``adapter_name`` (resolved in the ctor from ``backend.rollout_adapter_name``
        when not given) names which adapter's delta the walk reads — DiffusionNFT
        points it at the EMA ``"old"`` shadow so the rollout engine receives the
        EMA-smoothed weights.

        Each emitted name is then rewritten by ``name_remap`` (see
        ``_apply_name_remap``): a ``None`` rule drops the param (e.g. a frozen
        tower not in the receiver), and the ``"*"`` catch-all nests the trained
        submodule's bare keys under the receiver's namespace (e.g.
        ``transformer.``). It is orthogonal to LoRA-merging.

        ``wire_dtype`` (if set) is applied inside the walk, so the cast
        happens shard-side before the redistribute and every consumer
        (``_iter_buckets`` sizing included) sees wire-width tensors.
        """
        from unirl.utils.peft_merge import merged_state_dict, raw_state_dict

        remap = self._name_remap
        # Expert-parallel models: the stacked expert DTensor is sharded across the
        # ``ep`` group (global shape [E/ep] per rank), and the receiver (SGLang)
        # expects HF per-expert names. The raw walk would push only [E/ep] under
        # the fused name, so route EP experts through the EP-aware walk (all-gather
        # over ep + stacked->per-expert). Full-finetune (raw) path only: under
        # LoRA, experts are frozen and not re-synced. No-op when EP is disabled.
        if not self._lora_merged and getattr(self._backend.model, "_extra_parallel_param_groups", None) is not None:
            yield from self._iter_full_tensors_ep()
            return

        if self._lora_merged:
            for name, tensor in merged_state_dict(
                self._backend.model, adapter_name=self._adapter_name, dtype=self._wire_dtype
            ):
                out = _apply_name_remap(name, remap)
                if out is not None:
                    yield out, tensor
            return

        for name, tensor in raw_state_dict(
            self._backend.model, adapter_name=self._adapter_name, dtype=self._wire_dtype
        ):
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                continue
            out = _apply_name_remap(name, remap)
            if out is not None:
                yield out, tensor

    def _iter_full_tensors_ep(self) -> Iterator[Tuple[str, "object"]]:
        """EP-aware weight walk: emit HF per-expert tensors from the EP-sharded model.

        For each fused expert param (``...experts.gate_up_proj`` / ``down_proj``):
        materialize this rank's ``[E/ep, …]`` block (``_to_full_tensor`` replicates
        over ``ep_fsdp``), **all-gather across the ``ep`` group** to reconstruct the
        full ``[E, …]`` stack, then split it back into HF per-expert tensors
        (``experts.{e}.gate_proj`` = ``gate_up[e][:I]``, ``up_proj`` = ``[I:2I]``,
        ``down_proj`` = ``down[e]``) — the reverse of the load converter, the layout
        SGLang's Qwen3-MoE ``expert_params_mapping`` consumes. Non-expert params are
        emitted unchanged via ``_to_full_tensor``.

        Runs on every train rank in lockstep (the all-gather is collective). Each
        rank ends up with the full per-expert set and pushes it to its co-located
        engine, exactly like the dense BROADCAST sync.
        """
        import torch
        import torch.distributed as dist

        from unirl.train.backend.veomni import _compat
        from unirl.utils.peft_merge import _to_full_tensor

        _compat.ensure_installed()
        from veomni.distributed.parallel_state import get_parallel_state

        ps = get_parallel_state()
        ep_size = int(ps.ep_size) if getattr(ps, "ep_enabled", False) else 1
        ep_group = ps.ep_group if ep_size > 1 else None
        remap = self._name_remap

        for name, param in self._backend.model.state_dict().items():
            full = _to_full_tensor(param, self._wire_dtype)  # [E/ep,…] for experts; full otherwise

            is_gate_up = name.endswith(".experts.gate_up_proj")
            is_down = name.endswith(".experts.down_proj")
            if (is_gate_up or is_down) and ep_size > 1:
                local = full.contiguous()
                gathered = [torch.empty_like(local) for _ in range(ep_size)]
                dist.all_gather(gathered, local, group=ep_group)
                stacked = torch.cat(gathered, dim=0)  # [E, …]  (ep rank r -> experts r*E/ep …)
                del gathered, full, local
            elif is_gate_up or is_down:
                stacked = full  # ep_size==1: already the full [E,…]
            else:
                out = _apply_name_remap(name, remap)
                if out is not None:
                    yield out, full
                continue

            prefix = name.rsplit(".experts.", 1)[0]
            num_experts = int(stacked.shape[0])
            if is_gate_up:
                inter = stacked.shape[1] // 2
                for e in range(num_experts):
                    for nm, t in (
                        (f"{prefix}.experts.{e}.gate_proj.weight", stacked[e, :inter, :].contiguous()),
                        (f"{prefix}.experts.{e}.up_proj.weight", stacked[e, inter:, :].contiguous()),
                    ):
                        out = _apply_name_remap(nm, remap)
                        if out is not None:
                            yield out, t
            else:  # down_proj
                for e in range(num_experts):
                    out = _apply_name_remap(f"{prefix}.experts.{e}.down_proj.weight", remap)
                    if out is not None:
                        yield out, stacked[e].contiguous()
            del stacked

    def _iter_buckets(self) -> Iterator[Tuple[List[Tuple[str, "object"]], bool]]:
        """Yield ``(bucket, is_last)`` where ``bucket`` is a list of
        ``(name, tensor)`` up to ``bucket_size_mb``.

        ``is_last`` is True only for the final bucket — used to drive
        ``flush_cache`` on the receiver.
        """
        bucket: List[Tuple[str, object]] = []
        nbytes = 0
        for name, tensor in self._iter_full_tensors():
            size = tensor.numel() * tensor.element_size()
            if bucket and nbytes + size >= self._bucket_bytes:
                yield bucket, False
                bucket, nbytes = [], 0
            bucket.append((name, tensor))
            nbytes += size
        if bucket:
            yield bucket, True

    @property
    def _my_rank(self) -> int:
        ri = self.rank_info
        return ri.rank if ri is not None else 0


__all__ = ["FullWeightSync"]
