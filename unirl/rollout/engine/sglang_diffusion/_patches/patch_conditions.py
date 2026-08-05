"""Re-home the ``sglang-drl`` fork's text-encoder *conditions* emission (LIN-365).

UniRL's GRPO recipes that run ``populate_conditions=true`` consume
engine-emitted text-encoder embeddings: the response translator
(``rollout/engine/sglang/response.py:_build_text_conditions``) reads, per
``GenerationResult``::

    result.prompt_embeds, result.audio_prompt_embeds,
    result.pooled_prompt_embeds, result.encoder_attention_mask,
    result.negative_prompt_embeds, result.negative_audio_prompt_embeds,
    result.neg_pooled_prompt_embeds

Stock upstream ``GenerationResult`` / ``OutputBatch`` do NOT carry these
(fork-only), and upstream ``SamplingParams`` rejects ``return_prompt_embeds`` --
so the SD3 GRPO e2e crashes at
``SamplingParams.__init__() got an unexpected keyword argument
'return_prompt_embeds'``. This patch re-hosts the fork's conditions path on stock
upstream WITHOUT editing sglang source.

The flags themselves (``return_prompt_embeds`` / ``return_negative_prompt_embeds``)
are injected as ``SamplingParams`` fields by the sibling ``patch_sampling_io``
(see ``_SP_INJECT_FIELDS``); since ``Req`` has no such field, ``Req.__getattr__``
delegates the read to ``sampling_params``, so the worker sees them as
``result.return_prompt_embeds`` / ``result.return_negative_prompt_embeds``.

WHAT THIS PATCH DOES (all setattr / dataclass-field-injection / AROUND-wrap):

1. **OutputBatch + GenerationResult field injection.** Add the condition fields to
   each dataclass (mirrors the fork's schedule_batch.py / entrypoints/utils.py
   diffs) so they round-trip through ``dataclasses.fields`` / ``replace`` and the
   scheduler<->driver IPC.

2. **Copy the fields off the ``Req`` onto the OutputBatch, gated on the flags**,
   at the seam where the OutputBatch is actually built. In the MONOLITHIC path the
   terminal ``DecodingStage.forward(batch) -> OutputBatch`` constructs it directly,
   so ``GPUWorker._req_to_output_batch`` is bypassed (it fires only on the disagg
   raw-Req path) -- we therefore AROUND-wrap BOTH ``DecodingStage.forward`` (2a)
   and ``_req_to_output_batch`` (2b), sharing ``_copy_conditions``. Source-field
   mapping is the fork's (``gpu_worker.py`` OutputBatch construction diff)::

       prompt_embeds          <- result.prompt_embeds
       audio_prompt_embeds    <- result.audio_prompt_embeds
       pooled_prompt_embeds   <- result.pooled_embeds
       encoder_attention_mask <- result.prompt_embeds_mask
       negative_prompt_embeds <- result.negative_prompt_embeds
       negative_audio_prompt_embeds <- result.negative_audio_prompt_embeds
       neg_pooled_prompt_embeds <- result.neg_pooled_embeds
       negative_attention_mask  <- result.negative_prompt_embeds_mask

   Upstream's ``TextEncodingStage.forward`` ALREADY populates the positive batch
   fields (``prompt_embeds`` / ``pooled_embeds`` / ``prompt_embeds_mask`` -- the
   embeds-aligned mask the DiT actually attends under) and, when CFG is active, the
   negative ones (``negative_prompt_embeds`` / ``neg_pooled_embeds`` /
   ``negative_prompt_embeds_mask``) -- so we only COPY, never re-encode.
   That is why no text-encoding AROUND-wrap is needed here (see RISKS for why the
   fork's zeros-fallback / ``_expand`` re-capture is intentionally dropped).

3. **AROUND-wrap ``GPUWorker._merge_expanded_output_batches``** (the grouped
   nopp>1 path) to concat the per-output embed fields dim-0 onto the merged
   OutputBatch -- upstream's merge helpers do not carry them. No-op in the single
   path (that path never calls merge).

4. **AROUND-wrap ``DiffGenerator._result_common``** to copy the idx-th output's
   embed slice from the (single or merged) OutputBatch into the per-result
   GenerationResult kwargs. Slicing mirrors the fork's ``_slice_embed_list`` /
   upstream's ``samples_out[idx]`` per-output convention: each field is a
   ``list[Tensor]`` (one per text encoder), sliced ``t[idx:idx+1]`` so each
   GenerationResult carries its own single-sample embeds and the response
   translator's dim-0 concat over results reconstructs the batch.

Idempotent; setattr / field-injection / AROUND-wrap only -- no sglang source edits.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import field

logger = logging.getLogger(__name__)

# The condition fields default to None and are typed
# ``list[torch.Tensor] | None`` (one entry per text encoder) on
# OutputBatch; ``Any``-typed on GenerationResult to match its existing style.
#
# ``image_latent`` is a packed
# ``[1, S_img, C*4]`` tensor per output. It is represented as
# ``list[encoder=1][batch]`` so outputs with different ``S_img`` can coexist
# without padding or concatenating unlike token grids. Edit-Plus and other
# image-conditioned families such as FLUX.2 set ``batch.image_latent``;
# pure T2I requests leave it ``None``.
#
# ``image_latent_sizes`` (Edit-Plus only) carries each sample's
# ``vae_image_sizes`` (a ``list[tuple[int, int]]`` of pixel (W, H) pairs from
# upstream's ``preprocess_vae_image``). It is represented as
# ``list[encoder][batch][source_image]`` so grouped outputs and mixed-aspect
# prompts can be merged and sliced exactly like tensor conditions.
_COND_FIELDS = (
    "prompt_embeds",
    "audio_prompt_embeds",
    "pooled_prompt_embeds",
    "encoder_attention_mask",
    "negative_prompt_embeds",
    "negative_audio_prompt_embeds",
    "neg_pooled_prompt_embeds",
    "negative_attention_mask",
    "image_latent",
    "image_latent_sizes",
    "condition_image_latent_ids",
)

_POS_MAP = {
    "prompt_embeds": "prompt_embeds",
    "audio_prompt_embeds": "audio_prompt_embeds",
    "pooled_prompt_embeds": "pooled_embeds",
    "encoder_attention_mask": "prompt_embeds_mask",
}
_NEG_MAP = {
    "negative_prompt_embeds": "negative_prompt_embeds",
    "negative_audio_prompt_embeds": "negative_audio_prompt_embeds",
    "neg_pooled_prompt_embeds": "neg_pooled_embeds",
    "negative_attention_mask": "negative_prompt_embeds_mask",
}

_TOKEN_EMBED_DESTS = frozenset(
    {
        "prompt_embeds",
        "audio_prompt_embeds",
        "negative_prompt_embeds",
        "negative_audio_prompt_embeds",
    }
)

_OUTPUT_BATCH_FIELDS_SENTINEL = "_unirl_conditions_output_batch_fields"
_GEN_RESULT_FIELDS_SENTINEL = "_unirl_conditions_gen_result_fields"
_REQ_TO_OB_SENTINEL = "_unirl_conditions_req_to_ob"
_DECODING_SENTINEL = "_unirl_conditions_decoding"
_MERGE_SENTINEL = "_unirl_conditions_merge"
_RESULT_COMMON_SENTINEL = "_unirl_conditions_result_common"


def patch_conditions() -> None:
    """Install the fork's text-encoder conditions emission on stock upstream.

    Import-safe (all sglang imports are local) and idempotent.
    """
    import sglang.multimodal_gen.runtime.entrypoints.diffusion_generator as dg_mod
    import sglang.multimodal_gen.runtime.entrypoints.utils as utils_mod
    import sglang.multimodal_gen.runtime.managers.gpu_worker as gw_mod
    import sglang.multimodal_gen.runtime.pipelines_core.schedule_batch as sb_mod
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )

    _inject_dataclass_fields(
        sb_mod.OutputBatch,
        _OUTPUT_BATCH_FIELDS_SENTINEL,
        type_str="list[torch.Tensor] | None",
    )
    _inject_dataclass_fields(
        utils_mod.GenerationResult,
        _GEN_RESULT_FIELDS_SENTINEL,
        type_str="Any",
    )

    _wrap_decoding_stage(DecodingStage)

    _wrap_req_to_output_batch(gw_mod.GPUWorker)

    _wrap_merge_expanded_output_batches(gw_mod.GPUWorker)

    _wrap_result_common(dg_mod.DiffGenerator)


def _make_dataclass_field(name: str, default, type_str: str):
    """Build a ``dataclasses.Field`` equivalent to ``name: type = default``.

    Mirrors ``patch_sampling_io._make_dataclass_field``: registered as a real
    (init=True) field so ``dataclasses.fields`` / ``replace`` / ``asdict`` treat
    it like any source-declared field.
    """
    f = field(default=default)
    f.name = name
    f.type = type_str
    f._field_type = dataclasses._FIELD
    return f


def _inject_dataclass_fields(cls, sentinel: str, *, type_str: str) -> None:
    """Register the condition fields onto a plain ``@dataclass`` ``cls``.

    Registration (``__dataclass_fields__`` entry + class-level ``None`` default)
    makes the fields visible to ``dataclasses.fields`` / ``replace`` / ``asdict``,
    makes ``getattr(obj, name)`` return ``None`` pre-construction, and lets pickle
    round-trip them via ``__dict__``.

    The dataclass-generated ``__init__`` is frozen at class-creation time and does
    not know the post-hoc fields; yet once a field is in ``__dataclass_fields__``,
    ``dataclasses.replace`` passes EVERY field as a kwarg, and
    ``GenerationResult`` is built directly as ``GenerationResult(**common, ...)``
    with our keys. So we wrap ``__init__`` to strip the injected keys before the strict
    generated ``__init__`` runs, then re-apply via ``object.__setattr__`` -- the
    same strip-then-reapply pattern ``patch_sampling_io`` uses for SamplingParams.
    """
    if getattr(cls, sentinel, False):
        return

    own_fields = cls.__dict__.get("__dataclass_fields__")
    if own_fields is None:  # pragma: no cover - both are dataclasses
        own_fields = dict(getattr(cls, "__dataclass_fields__", {}))
        cls.__dataclass_fields__ = own_fields

    for name in _COND_FIELDS:
        if name not in own_fields:
            own_fields[name] = _make_dataclass_field(name, None, type_str)
        if name not in cls.__dict__:
            setattr(cls, name, None)

    orig_init = cls.__dict__.get("__init__")
    if orig_init is not None and not getattr(orig_init, sentinel, False):

        def __init__(self, *args, __orig_init=orig_init, **kwargs):
            extra = {k: kwargs.pop(k) for k in _COND_FIELDS if k in kwargs}
            __orig_init(self, *args, **kwargs)
            for k, v in extra.items():
                object.__setattr__(self, k, v)

        setattr(__init__, sentinel, True)
        cls.__init__ = __init__

    setattr(cls, sentinel, True)


def _wrap_req_to_output_batch(GPUWorker) -> None:
    """AROUND-wrap the ``@staticmethod`` Req -> OutputBatch conversion.

    Runs in both forward paths: ``_execute_forward_common`` (single) and
    ``_forward_group`` (grouped, per result before merge). Copies the embed
    fields off ``result`` (a ``Req``; reads delegate to ``sampling_params`` for
    the flags) onto the returned OutputBatch, gated on the flags. Verbatim source
    mapping from the fork's ``gpu_worker.py`` OutputBatch diff.
    """
    orig = GPUWorker.__dict__.get("_req_to_output_batch")
    if orig is None:
        raise AttributeError("GPUWorker._req_to_output_batch missing upstream")
    raw = orig.__func__ if isinstance(orig, staticmethod) else orig
    if getattr(raw, _REQ_TO_OB_SENTINEL, False):
        return

    def _req_to_output_batch(result):
        output_batch = raw(result)
        _copy_conditions(result, output_batch)
        return output_batch

    setattr(_req_to_output_batch, _REQ_TO_OB_SENTINEL, True)
    GPUWorker._req_to_output_batch = staticmethod(_req_to_output_batch)


def _copy_conditions(src, output_batch) -> None:
    """Copy the gated conditions fields off ``src`` (a Req) onto ``output_batch``.

    Shared by the decoding-stage wrap (monolithic path: the OutputBatch is built
    in ``DecodingStage.forward``) and ``_req_to_output_batch`` (disagg/raw-Req
    path). Source mapping is the fork's ``gpu_worker.py`` OutputBatch diff;
    positives gate on ``return_prompt_embeds``, negatives on
    ``return_negative_prompt_embeds`` (delegated to ``sampling_params``).
    """
    if getattr(src, "return_prompt_embeds", False):
        _copy_mapped_conditions(src, output_batch, _POS_MAP)
    if getattr(src, "return_negative_prompt_embeds", False):
        _copy_mapped_conditions(src, output_batch, _NEG_MAP)
    # Image-conditioned families expose packed [B, S_img, C] latents. Presence
    # on the batch is the gate; preserve a ragged batch axis
    # [encoder=1][batch] so mixed token counts do not require padding.
    image_latent = getattr(src, "image_latent", None)
    image_batch_size = None
    if image_latent is not None:
        import torch

        if torch.is_tensor(image_latent):
            rows = [row.detach().cpu() for row in image_latent.split(1, dim=0)]
        elif isinstance(image_latent, (list, tuple)):
            rows = []
            for latent in image_latent:
                if torch.is_tensor(latent):
                    rows.extend(row.detach().cpu() for row in latent.split(1, dim=0))
                else:
                    rows.append(latent)
        else:
            rows = [image_latent]
        image_batch_size = len(rows)
        output_batch.image_latent = [rows]
    # Edit-Plus vae_image_sizes: list[tuple[int, int]] of pixel (W, H) pairs
    # from upstream's preprocess_vae_image. Preserve an explicit batch axis:
    # [encoder=1][batch][source_image].
    vae_image_sizes = getattr(src, "vae_image_sizes", None)
    if vae_image_sizes is not None:
        per_sample_sizes = _normalize_vae_image_sizes(vae_image_sizes, image_batch_size)
        output_batch.image_latent_sizes = [per_sample_sizes]

    condition_image_latent_ids = getattr(src, "condition_image_latent_ids", None)
    if condition_image_latent_ids is not None:
        import torch

        if torch.is_tensor(condition_image_latent_ids):
            output_batch.condition_image_latent_ids = [condition_image_latent_ids.detach().cpu()]
        elif isinstance(condition_image_latent_ids, (list, tuple)):
            output_batch.condition_image_latent_ids = [
                t.detach().cpu() if torch.is_tensor(t) else t for t in condition_image_latent_ids
            ]


def _normalize_vae_image_sizes(vae_image_sizes, image_batch_size):
    """Normalize upstream size metadata to ``[batch][source_image]``.

    SGLang versions expose either a flat ``[(W, H)]`` list for one sample,
    one ``(W, H)`` tuple per batch row, or an already nested
    ``[batch][source_image]`` list. Reject every other shape here rather than
    letting the response adapter silently pair a latent with another sample's
    spatial grid.
    """

    if image_batch_size is None:
        raise ValueError("vae_image_sizes is present but image_latent is missing")
    if not isinstance(vae_image_sizes, (list, tuple)):
        raise TypeError(f"vae_image_sizes must be a list/tuple, got {type(vae_image_sizes).__name__}")

    entries = list(vae_image_sizes)
    if image_batch_size == 1 and entries and all(_is_image_size(size) for size in entries):
        # One sample with one or more source images.
        return [entries]
    if len(entries) != image_batch_size:
        raise ValueError(
            f"image_latent/vae_image_sizes batch mismatch: latents={image_batch_size}, sizes={len(entries)}"
        )
    if all(_is_image_size(size) for size in entries):
        # One source image per batch row.
        return [[size] for size in entries]
    if all(
        isinstance(sample_sizes, (list, tuple)) and sample_sizes and all(_is_image_size(size) for size in sample_sizes)
        for sample_sizes in entries
    ):
        return [list(sample_sizes) for sample_sizes in entries]
    raise ValueError(
        "vae_image_sizes must be [(W, H)] or [batch][source_image], "
        f"got {vae_image_sizes!r} for batch size {image_batch_size}"
    )


def _is_image_size(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(not isinstance(component, (list, tuple)) for component in value)
    )


def _copy_mapped_conditions(src, output_batch, mapping) -> None:
    """Copy each ``dst <- srcattr`` field, normalizing un-batched token embeds so
    every per-encoder field reaches the slice/merge transforms as ``[B, ...]``.

    Single-encoder token-level models (Z-Image) emit a bare ``[seq, hidden]``
    caption for a single-prompt encode; per-output ``_slice_embed_list`` would then
    slice the SEQ axis and corrupt it. Adding the missing batch dim here (gated to
    ``_TOKEN_EMBED_DESTS``) keeps every downstream transform batch-first; a no-op
    for already-batched multi-encoder embeds (SD3/Qwen) and for pooled/masks.
    """
    for dst, srcattr in mapping.items():
        val = _to_cpu_embed_list(getattr(src, srcattr, None))
        if dst in _TOKEN_EMBED_DESTS:
            val = _ensure_batched_embed_list(val)
            val = _coalesce_duplicate_single_sample_encodes(val)
        setattr(output_batch, dst, val)


def _ensure_batched_embed_list(value):
    """Add a leading batch dim to any un-batched ``[seq, hidden]`` per-encoder tensor.

    No-op for already-batched ``[B, seq, hidden]`` (``dim() >= 3``, multi-encoder
    models) and for ``None`` holes; preserves the container type.
    """
    if not isinstance(value, (list, tuple)):
        return value
    out = [t if (t is None or t.dim() >= 3) else t.unsqueeze(0) for t in value]
    return out if isinstance(value, list) else type(value)(out)


def _coalesce_duplicate_single_sample_encodes(value):
    """Collapse shallow-copy duplicate prompt encodes.

    Only same-shaped singleton-batch tensors ``[1, seq, hidden]`` are collapsed.
    Multi-encoder outputs (different shapes), non-tensors, and already-batched
    tensors are preserved.
    """
    import torch

    if not isinstance(value, (list, tuple)) or len(value) <= 1:
        return value
    if not all(torch.is_tensor(t) for t in value):
        return value

    first = value[0]
    first_shape = tuple(first.shape)
    if first.dim() < 1 or int(first.shape[0]) != 1:
        return value
    if any(tuple(t.shape) != first_shape for t in value[1:]):
        return value
    if not all(torch.equal(first, t) for t in value[1:]):
        return value
    return [first]


def _wrap_decoding_stage(DecodingStage) -> None:
    """AROUND-wrap ``DecodingStage.forward`` to carry conditions onto its OutputBatch.

    In the monolithic path the pipeline's terminal stage is decoding, whose
    ``forward(batch) -> OutputBatch`` (decoding.py) builds the OutputBatch directly
    from the ``batch`` Req -- so ``GPUWorker._req_to_output_batch`` (which only runs
    on the disagg raw-Req path) never fires, and the conditions never reach the
    OutputBatch. The ``batch`` Req still carries ``prompt_embeds`` (set by
    SD3ConditioningStage and untouched by timestep/latent/denoising), so copy them
    onto the returned OutputBatch here, gated on the flags. Runs per-output in the
    grouped path (``run_grouped_requests`` -> ``forward`` per Req).
    """
    orig = DecodingStage.__dict__.get("forward")
    if orig is None:
        raise AttributeError("DecodingStage.forward missing upstream")
    if getattr(orig, _DECODING_SENTINEL, False):
        return

    def forward(self, batch, server_args):
        output_batch = orig(self, batch, server_args)
        _copy_conditions(batch, output_batch)
        return output_batch

    setattr(forward, _DECODING_SENTINEL, True)
    DecodingStage.forward = forward


def _to_cpu_embed_list(value):
    """Detach + move a per-encoder ``list[Tensor]`` embed field to CPU.

    The OutputBatch is pickled across the scheduler<->driver ZMQ boundary; rollout
    tensors are materialized to CPU before transport (see
    ``rollout_denoising_mixin``'s ``.cpu()`` on ``dit_trajectory`` /
    ``rollout_log_probs``). Text-encoder embeds come off the batch on GPU, so we
    mirror that contract here -- otherwise a CUDA tensor would have to cross the
    process boundary (CUDA-IPC fragile / cross-device). The response translator
    reads them with ``.detach().cpu()`` so CPU here is exactly what it expects.

    Returns ``None`` unchanged; preserves a possible bare tensor (defensive --
    upstream stores these as lists per encoder) and per-element ``None`` holes.
    """
    if value is None:
        return None
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        moved = [t.detach().cpu() if torch.is_tensor(t) else t for t in value]
        return moved if isinstance(value, list) else type(value)(moved)
    return value


def _wrap_merge_expanded_output_batches(GPUWorker) -> None:
    """AROUND-wrap the grouped-output merge to carry conditions dim-0 concatenated.

    Upstream ``_merge_expanded_output_batches`` (and its collect/finalize helpers)
    does not carry the embed fields, so for an expanded ``num_outputs_per_prompt>1``
    request they would be dropped. We re-attach them by concatenating each
    field's per-encoder tensors across the per-output batches along dim-0, so the
    merged OutputBatch carries batch-dim-``N`` embeds that ``_result_common`` can
    then slice per output index.

    No-op in the single forward path -- that path returns the per-Req OutputBatch
    directly and never calls this method.
    """
    orig = GPUWorker.__dict__.get("_merge_expanded_output_batches")
    if orig is None:
        raise AttributeError("GPUWorker._merge_expanded_output_batches missing upstream")
    raw = orig.__func__ if isinstance(orig, staticmethod) else orig
    if getattr(raw, _MERGE_SENTINEL, False):
        return

    def _merge_expanded_output_batches(output_batches):
        merged = raw(output_batches)
        _merge_conditions(merged, output_batches)
        return merged

    setattr(_merge_expanded_output_batches, _MERGE_SENTINEL, True)
    GPUWorker._merge_expanded_output_batches = staticmethod(_merge_expanded_output_batches)


def _merge_conditions(merged, output_batches) -> None:
    """Concat each conditions field dim-0 across per-output batches onto ``merged``.

    Each field is ``list[Tensor]`` (per encoder); we concat the i-th encoder's
    tensor across all batches that carry it. If any batch is missing the field
    (None), the field is left None on ``merged`` -- positives are always present
    when ``return_prompt_embeds`` is set, negatives only under CFG.
    """
    import torch

    for name in _COND_FIELDS:
        per_batch = [getattr(ob, name, None) for ob in output_batches]
        if any(v is None for v in per_batch):
            continue
        if not per_batch:
            continue
        num_encoders = len(per_batch[0])
        # All batches must agree on encoder count to concat positionally.
        if any(len(v) != num_encoders for v in per_batch):
            logger.warning("conditions merge: inconsistent encoder count for %s; skipping", name)
            continue
        merged_list = []
        for enc_idx in range(num_encoders):
            tensors = [v[enc_idx] for v in per_batch]
            if any(t is None for t in tensors):
                merged_list.append(None)
            elif name == "image_latent":
                # Ragged [batch] tensors. Token counts may differ by aspect.
                merged_list.append(_merge_ragged_tensor_batches(tensors))
            elif name == "image_latent_sizes":
                # Ragged [batch][source_image] metadata. Extend only the batch
                # axis; keep each sample's source-image list intact.
                merged_list.append(_merge_image_size_batches(tensors))
            else:
                merged_list.append(torch.cat(tensors, dim=0))
        setattr(merged, name, merged_list)


def _merge_ragged_tensor_batches(values):
    """Flatten current ``[batch]`` and legacy dense batch representations."""
    import torch

    rows = []
    for value in values:
        if torch.is_tensor(value):
            rows.extend(value.split(1, dim=0))
        else:
            rows.extend(value)
    return rows


def _merge_image_size_batches(values):
    """Flatten only the batch axis of VAE image-size metadata."""
    merged = []
    for value in values:
        if value and all(_is_image_size(size) for size in value):
            # Legacy unbatched [source_image] metadata represents one sample.
            merged.append(list(value))
        else:
            merged.extend(list(sample_sizes) for sample_sizes in value)
    return merged


def _wrap_result_common(DiffGenerator) -> None:
    """AROUND-wrap ``DiffGenerator._result_common`` to add per-output embed slices.

    ``_result_common(req, output_batch, generation_time, output_index)`` returns
    the kwargs dict shared by every ``GenerationResult(**common, ...)`` call. We
    add the condition fields, slicing each per-encoder tensor ``t[idx:idx+1]``
    by ``output_index`` so each result carries its own single-sample embeds.

    Single path: ``output_batch`` is the per-Req batch (batch dim 1), idx=0 ->
    slice [0:1]. Grouped path: ``output_batch`` is the merged batch (batch dim N),
    idx in 0..N-1 -> slice [idx:idx+1]. The response translator concatenates over
    results (dim-0) to reconstruct the batch either way.
    """
    orig = DiffGenerator.__dict__.get("_result_common")
    if orig is None:
        raise AttributeError("DiffGenerator._result_common missing upstream")
    raw = orig.__func__ if isinstance(orig, staticmethod) else orig
    if getattr(raw, _RESULT_COMMON_SENTINEL, False):
        return

    def _result_common(req, output_batch, generation_time, output_index=None):
        common = raw(req, output_batch, generation_time, output_index)
        idx = 0 if output_index is None else int(output_index)
        for name in _COND_FIELDS:
            val = getattr(output_batch, name, None)
            if name == "image_latent":
                common[name] = _slice_ragged_tensor_list(val, idx)
            elif name == "image_latent_sizes":
                common[name] = _slice_image_size_list(val, idx)
            else:
                common[name] = _slice_embed_list(val, idx)
        return common

    setattr(_result_common, _RESULT_COMMON_SENTINEL, True)
    DiffGenerator._result_common = staticmethod(_result_common)


def _slice_embed_list(embed_list, idx: int):
    """Slice the idx-th sample out of a per-encoder ``list[Tensor]`` field.

    Returns a new list with each tensor sliced ``t[idx:idx+1]`` (keeps the batch
    dim), or ``None`` when the field is absent. Mirrors the fork's
    ``_slice_embed_list`` in ``diffusion_generator.py``.
    """
    if embed_list is None:
        return None
    return [t[idx : idx + 1] if t is not None else None for t in embed_list]


def _slice_image_size_list(size_list, idx: int):
    """Slice one output from ``[encoder][batch][source_image]`` metadata.

    Returns ``[encoder][source_image]``, the shape consumed by
    ``QwenImageEditPlusAdapter._collect_image_latents``. The legacy
    ``[encoder][source_image]`` shape is passed through for compatibility.
    """
    if size_list is None:
        return None
    sliced = []
    for per_encoder in size_list:
        if per_encoder is None:
            sliced.append(None)
            continue
        if per_encoder and all(_is_image_size(size) for size in per_encoder):
            if idx != 0:
                raise IndexError(
                    "legacy unbatched image_latent_sizes metadata only contains output index 0; "
                    f"cannot select index {idx}"
                )
            sliced.append(per_encoder)
            continue
        if idx >= len(per_encoder):
            raise IndexError(
                f"image_latent_sizes output index {idx} out of range for batch metadata of length {len(per_encoder)}"
            )
        sliced.append(per_encoder[idx])
    return sliced


def _slice_ragged_tensor_list(tensor_list, idx: int):
    """Slice one output from ``[encoder][batch]`` packed-latent metadata.

    Legacy ``[encoder][B,S,C]`` tensors are sliced on dim 0 for compatibility.
    """
    if tensor_list is None:
        return None
    sliced = []
    for per_encoder in tensor_list:
        if per_encoder is None:
            sliced.append(None)
        elif hasattr(per_encoder, "dim"):
            sliced.append(per_encoder[idx : idx + 1])
        else:
            if idx >= len(per_encoder):
                raise IndexError(
                    f"image_latent output index {idx} out of range for batch metadata of length {len(per_encoder)}"
                )
            sliced.append(per_encoder[idx])
    return sliced
