"""Shared sharded-weight loader for the FSDP and VeOmni backends.

Both backends materialize a meta-built, FSDP2-sharded trainable module and
broadcast real weights into it from rank 0.  The mechanics are identical — the
only backend difference is *timing*: VeOmni's ``parallelize`` already
``to_empty``-materializes the module before this runs, while FSDP's wrap leaves
it on meta.  :func:`_load_state_dict_sharded`'s meta-gate absorbs that
difference (it ``to_empty``s only if params are still on meta), so the same
loader is correct on both.

This module imports ``torch`` / ``safetensors`` at module level and MUST stay
out of the ``veomni`` package's import graph — it is imported only from inside
``backend.py`` — so the selective-import audit (``tests/test_compat_import.py``)
and the torch-free package-import check (``tests/test_recipe_compose.py``) stay
green.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict

import torch
from torch import nn

from unirl.train.backend.sharded_state import _build_state_dict_options, _current_rank

logger = logging.getLogger(__name__)

StateDict = Dict[str, object]


def load_trainable_weights(
    model: nn.Module,
    bundle: object,
    *,
    device: torch.device,
    rank: int = 0,
    with_aux: tuple[str, ...] = (),
    eager_ok: bool,
) -> None:
    """Resolve a bundle's trainable-weight source and load it post-wrap.

    Both backends call this immediately after wrapping the trainable module.
    Dispatch order:

    1. ``bundle._transformer_weights_path`` (meta-init "Pattern B"): load the
       stashed safetensors dir into the wrapped module via :func:`load_sharded`
       (its meta-gate ``to_empty``-materializes the still-meta FSDP module, then
       broadcasts).
    2. ``bundle.materialize(device, with_aux)`` (self-contained "Pattern A",
       e.g. hunyuan_image3): the bundle materializes itself.
    3. otherwise the bundle is eager — weights are already present. Tolerated
       when ``eager_ok`` (FSDP's wrap shards in place, leaving them intact); an
       error otherwise (VeOmni's ``parallelize`` already ``to_empty``'d the
       module, so eager weights would have been clobbered).
    """
    weights_path = getattr(bundle, "_transformer_weights_path", None)
    if weights_path is not None:
        load_sharded(model, weights_path, device=device, strict=False)
        # Recover init-computed non-persistent state (RoPE inv_freq, sincos tables,
        # …) clobbered by meta-init `to_empty` and not carried by the checkpoint.
        # The bundle carries the capture (capture_init_state); restoring here — in
        # the shared post-load path — is robust to the live trainer's Ray-actor
        # boundaries where the model-bound deferred closure can be dropped. Without
        # this the train model keeps garbage RoPE -> garbage replay log-probs ->
        # the DRPO rollout/replay ratio collapses (~0.05) and nothing learns.
        from unirl.models.types.meta_init import restore_init_state

        # Recover init-computed non-persistent buffers/attrs (RoPE inv_freq, sincos
        # tables, …) captured on the bundle before meta-init's to_empty clobbered them.
        n_recovered = restore_init_state(model, getattr(bundle, "_meta_init_state", None))
        # Re-establish TIED weights (lm_head <-> embed_tokens). For tie_word_embeddings
        # models, meta-init's to_empty breaks the tie and the checkpoint carries NO
        # separate lm_head.weight, so it stays uninitialized -> uniform logits ->
        # garbage replay log-probs (the DRPO rollout/replay ratio collapses to ~0.05
        # and nothing learns; SGLang ties its own lm_head so old_logp is fine).
        # tie_weights() re-points lm_head.weight at the loaded embed_tokens.weight.
        retied = False
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
            model.tie_weights()
            retied = True
        logger.info(
            "Rank %s: loaded trainable weights from %s (recovered %d non-persistent tensor(s), retied=%s)",
            rank,
            weights_path,
            n_recovered,
            retied,
        )
        return

    materialize = getattr(bundle, "materialize", None)
    if callable(materialize):
        materialize(device=device, with_aux=tuple(with_aux))
        return

    if not eager_ok:
        raise ValueError(
            "sharded_load: trainable module has no weight source — a meta-init "
            "bundle must stash `_transformer_weights_path` or provide "
            "materialize(). Eagerly-loaded bundles are FSDP-only: this backend's "
            "parallelize already materialized (to_empty) the module, so eager "
            "weights would be clobbered."
        )
    if with_aux:
        logger.info(
            "Rank %s: bundle %s loads eagerly; ignoring with_aux=%s",
            rank,
            type(bundle).__name__,
            tuple(with_aux),
        )


def load_sharded(
    module: nn.Module,
    weights_dir: str,
    *,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Materialize ``module`` from a (diffusers-layout) safetensors directory.

    Rank 0 reads every ``*.safetensors`` shard under ``weights_dir``; the
    weights are broadcast into the sharded module.  See
    :func:`_load_state_dict_sharded` for the per-rank mechanics.  This is the
    common path for single-module trainables whose weights live in a dedicated
    directory (diffusion ``<ckpt>/transformer``, AR ``<ckpt>`` root).
    """
    # Expert-parallel models shard stacked expert weights on a 2D composed mesh
    # (``ep_fsdp x ep``); torch's rank-0-broadcast loader mis-slices that layout.
    # Each rank reads ONLY its own expert block from the (local) safetensors via
    # mmap'd ``get_slice`` and uses DTensor-native ``distribute_tensor`` to fill
    # its exact local shard. Non-EP models keep the rank-0-broadcast path verbatim.
    if hasattr(module, "_extra_parallel_param_groups"):
        if _module_has_meta_param(module):
            module.to_empty(device=device)
        _load_state_dict_ep_sliced(module, weights_dir, device=device, strict=strict)
        return

    state_dict = _read_safetensors_dir(weights_dir) if _current_rank() == 0 else {}
    _load_state_dict_sharded(module, state_dict, device=device, strict=strict)


def _build_expert_block_from_split(name, dst, key_to_handle, ckpt_keys, ep_size, ep_rank, device):
    """Reconstruct THIS ep rank's fused expert block from HF per-expert keys.

    Mirrors VeOmni's ``Qwen3MoeCheckpointTensorConverter`` (per-expert HF ->
    fused v5), but applied **only to the experts this rank owns** so memory stays
    at ``E/ep``:

        ``{prefix}.experts.{e}.gate_proj.weight`` [I,H] + ``up_proj.weight`` [I,H]
            -> per expert ``cat([gate, up], dim=0)`` [2I,H], stacked  -> [E/ep, 2I, H]
        ``{prefix}.experts.{e}.down_proj.weight`` [H,I]  stacked        -> [E/ep, H, I]

    Returns the block tensor on ``device`` (dtype ``dst.dtype``), or ``None`` when
    ``name`` is not a fused expert param or the per-expert keys are absent (so the
    caller falls through to the normal missing-key handling).
    """
    if name.endswith(".experts.gate_up_proj"):
        proj = "gate_up_proj"
    elif name.endswith(".experts.down_proj"):
        proj = "down_proj"
    else:
        return None

    prefix = name.rsplit(".experts.", 1)[0]  # "...mlp"
    num_local = int(tuple(dst.shape)[0])  # E/ep (the param's global dim-0)
    start = ep_rank * num_local
    experts = range(start, start + num_local)

    def _resolve(keys: list) -> "list | None":
        """None if NO key present (not split format -> caller handles missing);
        raise if SOME but not all present (corrupt/incomplete checkpoint)."""
        present = [k for k in keys if k in ckpt_keys]
        if not present:
            return None
        if len(present) != len(keys):
            raise RuntimeError(
                f"EP load: incomplete per-expert checkpoint for {name!r}: "
                f"{len(present)}/{len(keys)} keys present (e.g. missing "
                f"{next(k for k in keys if k not in ckpt_keys)!r})."
            )
        return keys

    if proj == "down_proj":
        keys = _resolve([f"{prefix}.experts.{e}.down_proj.weight" for e in experts])
        if keys is None:
            return None
        block = torch.stack([key_to_handle[k].get_tensor(k) for k in keys])  # [E/ep, H, I]
    else:
        gkeys = _resolve([f"{prefix}.experts.{e}.gate_proj.weight" for e in experts])
        ukeys = _resolve([f"{prefix}.experts.{e}.up_proj.weight" for e in experts])
        if gkeys is None or ukeys is None:
            return None
        per_expert = [
            torch.cat([key_to_handle[g].get_tensor(g), key_to_handle[u].get_tensor(u)], dim=0)  # [2I, H]
            for g, u in zip(gkeys, ukeys)
        ]
        block = torch.stack(per_expert)  # [E/ep, 2I, H]

    if tuple(block.shape) != tuple(dst.shape):
        raise RuntimeError(
            f"EP load: rebuilt expert block for {name!r} has shape {tuple(block.shape)} "
            f"!= param global shape {tuple(dst.shape)} (ep_rank={ep_rank}, num_local={num_local})."
        )
    return block.to(device=device, dtype=dst.dtype)


def _load_state_dict_ep_sliced(
    module: nn.Module,
    weights_dir: str,
    *,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Memory-optimal EP weight load: each rank reads ONLY its expert block.

    The EP plan pre-slices each expert param along the expert dim, so the
    DTensor's GLOBAL shape is already ``[E/ep, …]`` (the ``ep`` split is baked in
    per rank, NOT a DTensor placement). The checkpoint stores the full
    ``[E, …]``. Instead of materializing the full tensor on every rank (8x host
    RAM) or broadcasting it from rank 0 (rank-0 full-RAM bottleneck), we
    ``safetensors.get_slice`` the file (mmap, lazy) and index **only this ep
    rank's contiguous ``[E/ep, …]`` byte range** — no rank ever holds the full
    ``[E, …]`` and there is no cross-rank broadcast. ``distribute_tensor`` then
    shards that block across the remaining ``ep_fsdp`` FSDP dim. Non-expert params
    (global == checkpoint shape) are read whole (they are small: embed / norm /
    lm_head) and distributed over their own mesh.

    Strictly dominates the full-read and rank-0-broadcast variants on peak host
    RAM (``E/ep`` vs full) and avoids broadcast traffic; the per-rank reads run in
    parallel off the shared file (shared mmap page cache on one node).

    Checkpoint formats: both VeOmni **stacked** (``experts.gate_up_proj`` /
    ``down_proj``) and HF **original per-expert** (``experts.{e}.gate_proj`` /
    ``up_proj`` / ``down_proj``) are accepted — the latter is reconstructed per
    rank via :func:`_build_expert_block_from_split` (VeOmni's converter mapping),
    so real Qwen3-MoE HF checkpoints load directly with no offline merge.
    """
    import glob

    from safetensors import safe_open
    from torch.distributed.tensor import DTensor, distribute_tensor

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    ep_size = int(ps.ep_size) if getattr(ps, "ep_enabled", False) else 1
    ep_rank = int(ps.ep_rank) if ep_size > 1 else 0

    shards = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"EP load: no *.safetensors under {weights_dir!r}")
    # Open each shard once (mmap; header read only) and map key -> handle.
    handles = {s: safe_open(s, framework="pt", device="cpu") for s in shards}
    key_to_handle = {}
    for s, h in handles.items():
        for k in h.keys():
            key_to_handle[k] = h
    ckpt_keys = set(key_to_handle)

    named = dict(module.named_parameters())
    named.update(dict(module.named_buffers()))

    missing, loaded, local_elems = [], 0, 0
    for name, dst in named.items():
        # LoRA inserts a ``base_layer`` hop; the base checkpoint omits it.
        ckpt_key = name
        if ckpt_key not in ckpt_keys:
            stem, _, leaf = name.rpartition(".")
            cand = f"{stem.removesuffix('.base_layer')}.{leaf}" if stem.endswith(".base_layer") else name
            ckpt_key = cand if cand in ckpt_keys else name
        if ckpt_key not in ckpt_keys:
            # HF original (per-expert split) checkpoint: the fused expert param
            # (``...experts.gate_up_proj`` / ``...experts.down_proj``) is absent;
            # rebuild THIS ep rank's block from the per-expert ``experts.{e}.*``
            # keys (VeOmni's CheckpointTensorConverter mapping, applied per rank).
            block = _build_expert_block_from_split(name, dst, key_to_handle, ckpt_keys, ep_size, ep_rank, device)
            if block is not None:
                local_elems += block.numel()
                if isinstance(dst, DTensor):
                    sharded = distribute_tensor(block, dst.device_mesh, dst.placements)
                    with torch.no_grad():
                        dst.to_local().copy_(sharded.to_local())
                else:
                    with torch.no_grad():
                        dst.copy_(block)
                loaded += 1
                del block
                continue
            missing.append(name)
            continue

        sl = key_to_handle[ckpt_key].get_slice(ckpt_key)
        ckpt_shape = tuple(sl.get_shape())
        global_shape = tuple(dst.shape)

        # EP-presliced experts: the fused expert param's global shape is [E/ep,…]
        # while the (stacked) checkpoint holds [E,…]. Only these params are
        # ep-presliced, so gate the dim-0 slice on the expert-param name — never
        # infer it from a coincidental ``ckpt==global*ep_size`` shape match on some
        # other tensor. Slice this rank's contiguous expert block on dim 0.
        is_expert = name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj")
        if is_expert and ckpt_shape != global_shape and ep_size > 1:
            if ckpt_shape[0] != global_shape[0] * ep_size or ckpt_shape[1:] != global_shape[1:]:
                raise RuntimeError(
                    f"EP load: expert param {name!r} shape mismatch: checkpoint {ckpt_shape} "
                    f"vs expected [E={global_shape[0] * ep_size} on dim0, {global_shape[1:]}]."
                )
            n = global_shape[0]
            block = sl[ep_rank * n : (ep_rank + 1) * n]  # only this rank's E/ep experts
            if tuple(block.shape) != global_shape:
                raise RuntimeError(f"EP load: sliced {name!r} to {tuple(block.shape)} != {global_shape}.")
        elif ckpt_shape != global_shape:
            # Non-expert param whose checkpoint shape disagrees with the sharded
            # param global shape — a real mismatch, not an ep preslice.
            raise RuntimeError(
                f"EP load: shape mismatch for non-expert param {name!r}: checkpoint {ckpt_shape} vs global {global_shape}."
            )
        else:
            block = sl[:]  # whole (small) tensor; global == checkpoint shape

        block = block.to(device=device, dtype=dst.dtype)
        local_elems += block.numel()
        if isinstance(dst, DTensor):
            sharded = distribute_tensor(block, dst.device_mesh, dst.placements)
            with torch.no_grad():
                dst.to_local().copy_(sharded.to_local())
        else:
            with torch.no_grad():
                dst.copy_(block)
        loaded += 1
        del block

    del handles, key_to_handle  # close mmaps

    if strict and missing:
        raise RuntimeError(f"EP load: missing {len(missing)} tensor(s) in checkpoint: {missing[:8]}")
    if missing and _current_rank() == 0:
        logger.info(
            "EP load: %d tensor(s) absent from checkpoint (e.g. non-persistent buffers): %s%s",
            len(missing),
            missing[:6],
            " ..." if len(missing) > 6 else "",
        )
    if _current_rank() == 0:
        logger.info(
            "EP load: loaded %d tensor(s) via get_slice+distribute_tensor (rank0 read %.0fM elems)",
            loaded,
            local_elems / 1e6,
        )


def _load_state_dict_sharded(
    module: nn.Module,
    state_dict: StateDict,
    *,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Allocate storage for any meta params, then broadcast-load ``state_dict``.

    ``state_dict`` is the rank-0 full state dict (empty ``{}`` on other ranks).
    Steps:

    1. ``to_empty(device)`` any submodule still on meta — gated, so it is a
       no-op when the wrap already materialized the module (VeOmni's
       ``parallelize``) and the allocator that the FSDP-meta path needs when it
       did not.
    2. rank 0: insert the ``base_layer`` hop for LoRA-injected modules.
    3. ``set_model_state_dict(..., broadcast_from_rank0=True, strict=strict)``
       — DTensor-aware; handles FSDP2 shards + plain params in one collective.
    """
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    if _module_has_meta_param(module):
        module.to_empty(device=device)

    if _current_rank() == 0:
        state_dict = _remap_lora_base_keys(state_dict, module)

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
        strict=strict,
    )
    try:
        set_model_state_dict(module, state_dict, options=options)
    except TypeError:
        set_model_state_dict(module, state_dict)


def _module_has_meta_param(module: nn.Module) -> bool:
    """True if any parameter of ``module`` (recursing into children) is on the
    meta device.  Used to gate the per-shard ``to_empty`` call."""
    return any(p.is_meta for p in module.parameters(recurse=True))


def _read_safetensors_dir(weights_dir: str) -> StateDict:
    """Merge all ``*.safetensors`` shards in a directory.

    Loading every shard makes the index json unnecessary and covers both
    single-file and sharded checkpoints."""
    from safetensors.torch import load_file

    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"sharded_load: transformer weights dir not found: {weights_dir!r}. "
            "HF repo IDs are not supported here — point the recipe's checkpoint "
            "path at a local download."
        )
    shards = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"sharded_load: no *.safetensors files under {weights_dir!r}")
    state_dict: StateDict = {}
    for shard in shards:
        state_dict.update(load_file(shard, device="cpu"))
    return state_dict


def _remap_lora_base_keys(state_dict: StateDict, model: nn.Module) -> StateDict:
    """Translate base-checkpoint keys for LoRA-injected modules.

    ``peft.inject_adapter_in_model`` (via ``unirl.train.lora`` /
    ``unirl.train.ema``) rewires target Linears in place, so their original
    weight moves to ``<module>.base_layer.weight``.  The base checkpoint still
    uses the original key — insert the ``base_layer`` hop where (and only
    where) the model expects it."""
    model_keys = {n for n, _ in model.named_parameters()}
    model_keys.update(n for n, _ in model.named_buffers())
    remapped: StateDict = {}
    for key, value in state_dict.items():
        if key not in model_keys:
            stem, _, leaf = key.rpartition(".")
            candidate = f"{stem}.base_layer.{leaf}" if stem else key
            if candidate in model_keys:
                remapped[candidate] = value
                continue
        remapped[key] = value
    return remapped


__all__ = ["load_trainable_weights", "load_sharded", "_load_state_dict_sharded", "StateDict"]
