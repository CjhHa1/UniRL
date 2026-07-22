"""Qwen3MoeBundle — VeOmni-patched Qwen3-MoE causal LM + tokenizer.

Unlike :class:`unirl.models.qwen3.Qwen3Bundle` (HF ``AutoModelForCausalLM``,
EP-incapable), this builds the model through ``veomni.build_foundation_model``
so it carries:

* ``get_parallel_plan`` — ``Shard(0)`` on the stacked expert weights
  (``mlp.experts.gate_up_proj`` / ``down_proj``), which
  ``VeOmniBackend`` (``fsdp_cfg.ep_size>1``) shards across the EP submesh; and
* a fused MoE op (``moe_implementation="fused_triton"``) whose forward runs the
  all-to-all dispatch / grouped-GEMM / all-to-all combine EP path.

Meta-init only (``VeOmniBackend`` materializes via ``to_empty`` + loads the
stacked safetensors after sharding). The model's non-persistent RoPE
``inv_freq`` buffers (absent from the checkpoint, clobbered by ``to_empty``) are
recomputed from config by a stamped deferred op drained post-load by
``apply_deferred_ops``.

The dense Qwen3 ``Qwen3ARStage`` / ``Qwen3ARConditions`` are reused verbatim —
the replay forward only needs ``.model`` (decoder) + ``.lm_head``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.train.deferred import _stamp
from unirl.utils.dtypes import canonical_torch_dtype_name, parse_torch_dtype

logger = logging.getLogger(__name__)


def _recompute_rope_buffers(model: nn.Module) -> int:
    """Recompute non-persistent RoPE ``inv_freq`` from config on the materialized
    model (``to_empty`` left them as garbage; the checkpoint does not carry them).
    Best-effort: matches any module exposing ``inv_freq`` + ``config``."""
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return 0
    n = 0
    device = next(model.parameters()).device
    for m in model.modules():
        inv = getattr(m, "inv_freq", None)
        cfg = getattr(m, "config", None)
        if inv is None or cfg is None:
            continue
        fn = ROPE_INIT_FUNCTIONS.get(getattr(m, "rope_type", "default")) or ROPE_INIT_FUNCTIONS.get("default")
        if fn is None:
            continue
        try:
            inv_freq, scaling = fn(cfg, device)
        except Exception:
            continue
        with torch.no_grad():
            # After veomni_parallelize (FSDP2 fully_shard), these non-persistent
            # buffers can be DTensors. A plain ``buf.copy_(plain_tensor)`` onto a
            # DTensor raises (mixed DTensor/plain) and was swallowed by the caller's
            # try/except — leaving the ``to_empty`` ZEROS. inv_freq==0 makes RoPE the
            # identity (cos=1,sin=0 at every position) => a position-blind model =>
            # replay logprobs systematically wrong => rollout/replay ratio ~0.11 =>
            # GRPO fully clipped => reward can't move. Copy into the LOCAL shard.
            for _bn in ("inv_freq", "original_inv_freq"):
                _b = getattr(m, _bn, None)
                if _b is None:
                    continue
                _t = _b.to_local() if hasattr(_b, "to_local") else _b
                _t.copy_(inv_freq.to(device=_t.device, dtype=_t.dtype))
        m.attention_scaling = scaling
        n += 1
    if n:
        logger.info("Qwen3MoeBundle: recomputed RoPE inv_freq on %d rotary module(s)", n)
    return n


def _capture_rope_init_state(model: nn.Module) -> dict:
    """Compute the non-persistent RoPE ``inv_freq`` from config as a picklable
    ``_meta_init_state`` (``{"buffers": {fqn: cpu_tensor}, "attrs": {}}``), the
    robust bundle-carried recovery path ``load_trainable_weights`` ->
    ``restore_init_state`` drains after the post-shard weight load.

    Unlike a model-bound deferred closure (which the meta-init code documents can
    be dropped when the bundle crosses Ray actors), these are plain CPU tensors on
    the bundle, so the recovery survives transport. ``inv_freq`` depends only on
    config, so it is recomputable here on the meta model (``m.config`` /
    ``m.rope_type`` are real Python attrs; only the buffer storage is on meta).
    ``attention_scaling`` is a plain float attribute untouched by ``to_empty``, so
    it needs no restore.
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return {"buffers": {}, "attrs": {}}
    buffers: dict = {}
    for name, m in model.named_modules():
        if getattr(m, "inv_freq", None) is None or getattr(m, "config", None) is None:
            continue
        fn = ROPE_INIT_FUNCTIONS.get(getattr(m, "rope_type", "default")) or ROPE_INIT_FUNCTIONS.get("default")
        if fn is None:
            continue
        try:
            inv_freq, _scaling = fn(m.config, torch.device("cpu"))
        except Exception:
            continue
        inv_freq = inv_freq.detach().cpu()
        buffers[f"{name}.inv_freq"] = inv_freq
        if getattr(m, "original_inv_freq", None) is not None:
            buffers[f"{name}.original_inv_freq"] = inv_freq.clone()
    return {"buffers": buffers, "attrs": {}}


class Qwen3MoeBundle(Bundle):
    """VeOmni Qwen3-MoE transformer + tokenizer (meta-init, EP-capable)."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    def prepare_for_expert_parallel(self) -> None:
        """Backend hook when ``ep_size > 1``.

        VeOmni's ``build_foundation_model(..., moe_implementation=fused_triton)``
        already installs fused stacked experts and ``get_parallel_plan``, so
        unlike HI3 there is nothing to swap on meta — this is intentionally a
        no-op that satisfies :meth:`VeOmniBackend`'s EP-ready contract.
        """
        if not callable(getattr(self.transformer, "get_parallel_plan", None)):
            raise ValueError(
                "Qwen3MoeBundle.prepare_for_expert_parallel: transformer lacks "
                "get_parallel_plan(); rebuild with moe_implementation=fused_triton"
            )

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        *,
        pretrained_model_ckpt_path: Optional[str] = None,
        tokenizer_ckpt_path: Optional[str] = None,
        model_precision: str = "bf16",
        moe_implementation: str = "fused_triton",
        attn_implementation: str = "flash_attention_2",
        tokenizer: Any = None,
    ) -> "Qwen3MoeBundle":
        """Build the VeOmni Qwen3-MoE transformer (on meta) + tokenizer.

        Two call styles:
        * recipe (hydra): ``from_config(config=Qwen3PipelineConfig(...))`` — reads
          ``pretrained_model_ckpt_path`` / ``tokenizer_ckpt_path`` / ``model_precision``
          / ``attn_implementation`` off the config; ``moe_implementation`` falls back
          to ``"fused_triton"`` when the config does not carry it;
        * direct: ``from_config(pretrained_model_ckpt_path=..., tokenizer=...)``.

        ``pretrained_model_ckpt_path`` is a local dir with ``config.json`` and
        ``*.safetensors``. Both layouts load directly (no offline merge): VeOmni
        **stacked** (``experts.gate_up_proj`` / ``down_proj``) and HF **original
        per-expert** (``experts.N.gate_proj`` / ``up_proj`` / ``down_proj``) — the
        EP-aware loader (``unirl.train.backend.sharded_load``) reconstructs each
        rank's fused expert block from the per-expert keys when needed.
        """
        if config is not None:
            pretrained_model_ckpt_path = config.pretrained_model_ckpt_path
            tokenizer_ckpt_path = getattr(config, "tokenizer_ckpt_path", None)
            model_precision = getattr(config, "model_precision", "bf16")
            attn_implementation = getattr(config, "attn_implementation", None) or attn_implementation
            moe_implementation = getattr(config, "moe_implementation", None) or moe_implementation
        if pretrained_model_ckpt_path is None:
            raise ValueError("Qwen3MoeBundle.from_config: pretrained_model_ckpt_path is required")

        from unirl.train.backend.veomni import _compat

        _compat.ensure_installed()
        from veomni.arguments import OpsImplementationConfig
        from veomni.models.auto import build_foundation_model

        dtype = parse_torch_dtype(str(model_precision), field_name="Qwen3MoeBundle.model_precision")
        dtype_name = canonical_torch_dtype_name(dtype, field_name="Qwen3MoeBundle.model_precision")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ops = OpsImplementationConfig(
            attn_implementation=attn_implementation,
            moe_implementation=moe_implementation,
            cross_entropy_loss_implementation="eager",
            rms_norm_implementation="eager",
            swiglu_mlp_implementation="eager",
            rotary_pos_emb_implementation="eager",
            load_balancing_loss_implementation="eager",
        )
        transformer = build_foundation_model(
            config_path=pretrained_model_ckpt_path,
            weights_path=None,
            torch_dtype=dtype_name,
            init_device="meta",
            ops_implementation=ops,
        )
        # VeOmni's parallelize calls init_weights() after to_empty; no-op it (real
        # weights load right after, via the backend's EP-aware sharded load).
        transformer.init_weights = lambda: None
        # RoPE inv_freq is non-persistent (absent from the checkpoint, clobbered by
        # to_empty). Recover it via BOTH:
        #  * the bundle-carried _meta_init_state (ROBUST): plain CPU tensors that
        #    survive Ray-actor transport; load_trainable_weights -> restore_init_state
        #    copies them into the materialized buffers after the weight load.
        #  * a stamped in-process closure (belt): drained by apply_deferred_ops for
        #    any path that skips the bundle-carried restore. Both are idempotent.
        rope_state = _capture_rope_init_state(transformer)
        _stamp(transformer, lambda materialized: _recompute_rope_buffers(materialized))

        if tokenizer is None:
            from transformers import AutoTokenizer

            tok_path = tokenizer_ckpt_path or pretrained_model_ckpt_path
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        bundle = cls(
            transformer=transformer,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=pretrained_model_ckpt_path,
        )
        # Meta-init Pattern B: backend loads stacked safetensors from this dir
        # after EP sharding.
        bundle._transformer_weights_path = pretrained_model_ckpt_path
        # Ray-safe RoPE recovery (see _capture_rope_init_state): restored by
        # load_trainable_weights after the weight load.
        bundle._meta_init_state = rope_state
        return bundle


__all__ = ["Qwen3MoeBundle"]
