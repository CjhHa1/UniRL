#!/usr/bin/env python
"""Numerical validation + stage-level timing for SD3 batched-step replay.

The PE diffusion train phase replays the rollout's SDE transitions through the
SD3 transformer ONE step at a time (a serial loop in SD3DiffusionStage.replay).
``batch_replay_steps=True`` stacks all S steps on the batch dim and runs ONE
forward + one vectorized SDE transition. This script proves the batched path is
correct and ratio-preserving, on a single GPU (transformer only, no FSDP/Ray):

  Claim 1 (parity): batched per-(sample,step) log-prob == serial within bf16
    batch-shape tolerance, with the [B,S] mapping intact (no scramble).
  Claim 2 (ratio=1): batched replay is deterministic under no_grad -> two
    replays give bit-identical log-probs, so with old_logp_source='replay' the
    PPO ratio is exactly 1 (the anchor is replayed through the same batched
    path). This is the on-policy invariant the recipe needs.
  Claim 3 (grad): backward flows through the batched replay (train path works).

Also times serial vs batched replay (fwd and fwd+bwd) at the PE micro geometry.

Run (occupier neutered):
  CUDA_VISIBLE_DEVICES=0 SD3_DIR=/data/models/stable-diffusion-3.5-medium \
    python scripts/profiling/validate_batched_replay.py
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import torch

from unirl.models.sd3.conditions import SD3Conditions
from unirl.models.sd3.diffusion import SD3DiffusionStage, SD3DiffusionStep
from unirl.sde.kernels import FlowSDEStrategy
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.conditions import TextEmbedCondition
from unirl.types.sampling import DiffusionSamplingParams

SD3_DIR = os.environ.get("SD3_DIR", "/data/models/stable-diffusion-3.5-medium")
B = int(os.environ.get("B", "4"))
T = int(os.environ.get("T", "10"))
SDE_INDICES = [int(i) for i in os.environ.get("SDE_INDICES", "0,2,4").split(",")]
SHIFT = 3.0
DTYPE = torch.bfloat16


def _build_stage(batch_replay_steps: bool) -> SD3DiffusionStage:
    from diffusers import SD3Transformer2DModel

    tf = SD3Transformer2DModel.from_pretrained(f"{SD3_DIR}/transformer", torch_dtype=DTYPE).to("cuda").eval()
    bundle = SimpleNamespace(transformer=tf, device=torch.device("cuda"))
    return SD3DiffusionStage(
        model=bundle,
        step=SD3DiffusionStep(),
        strategy=FlowSDEStrategy(),
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        batch_replay_steps=batch_replay_steps,
    )


def _conditions(seed: int = 0) -> SD3Conditions:
    g = torch.Generator(device="cuda").manual_seed(seed)
    embeds = torch.randn(B, 333, 4096, device="cuda", dtype=DTYPE, generator=g)
    pooled = torch.randn(B, 2048, device="cuda", dtype=DTYPE, generator=g)
    return SD3Conditions(text=TextEmbedCondition(embeds=embeds, pooled=pooled), negative_text=None)


def main() -> None:
    torch.manual_seed(0)
    print(f"[validate] loading SD3 transformer (B={B}, T={T}, sde={SDE_INDICES}) ...", flush=True)
    stage = _build_stage(batch_replay_steps=False)
    stage_b = SD3DiffusionStage(  # share the SAME transformer weights, batched flag on
        model=stage.model,
        step=SD3DiffusionStep(),
        strategy=FlowSDEStrategy(),
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        batch_replay_steps=True,
    )

    schedule = get_sigma_schedule(T, shift=SHIFT, device=torch.device("cuda"))
    params = DiffusionSamplingParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=512,
        width=512,
        eta=0.7,
        seed=0,
        sde_indices=list(SDE_INDICES),
    )
    conds = _conditions()
    x0 = torch.randn(B, 16, 64, 64, device="cuda", dtype=stage.trajectory_dtype)

    # Build a real stored trajectory (diffuse uses the serial path; the stored
    # latents + sde_logp are what replay consumes).
    with torch.no_grad():
        seg = stage.diffuse(conds, schedule=schedule, params=params, initial_latents=x0)
    print(
        f"[validate] segment latents={tuple(seg.latents.shape)} sde_indices={seg.sde_indices.tolist()} "
        f"sde_logp={tuple(seg.sde_logp.shape)}",
        flush=True,
    )

    # ---- Claim 1: batched vs serial replay parity ----
    with torch.no_grad():
        rep_s = stage.replay(conds, segment=seg, params=params)
        rep_b = stage_b.replay(conds, segment=seg, params=params)
    lp_s, lp_b = rep_s.log_probs.float(), rep_b.log_probs.float()
    assert lp_s.shape == lp_b.shape == (B, len(SDE_INDICES)), (lp_s.shape, lp_b.shape)
    abs_diff = (lp_s - lp_b).abs()
    rel = abs_diff / (lp_s.abs() + 1e-6)
    print("\n[Claim 1] batched vs serial replay log-probs (per-sample logp ~ O(1e3..1e4)):", flush=True)
    print(
        f"  shape={tuple(lp_b.shape)}  max|abs diff|={abs_diff.max().item():.4e}  "
        f"max rel={rel.max().item():.4e}  mean rel={rel.mean().item():.4e}",
        flush=True,
    )
    # bf16 batch-shape rounding on a sum over ~64k latent elements: a few e-3 relative.
    ok1 = rel.max().item() < 2e-2
    # Mapping intact: diagonal (matched) rel must be << cross-sample rel.
    cross = ((lp_s[0] - lp_b[(0 + 1) % B]).abs() / (lp_s[0].abs() + 1e-6)).mean().item()
    print(
        f"  matched mean rel={rel.mean().item():.4e}  vs cross-sample rel={cross:.4e} "
        f"(cross must be >> matched -> [B,S] mapping not scrambled)",
        flush=True,
    )
    ok1 = ok1 and (cross > 10 * rel.mean().item())

    # ---- Claim 2: batched replay deterministic -> ratio=1 ----
    with torch.no_grad():
        r1 = stage_b.replay(conds, segment=seg, params=params).log_probs.float()
        r2 = stage_b.replay(conds, segment=seg, params=params).log_probs.float()
    max_ratio_dev = (torch.exp(r1 - r2) - 1.0).abs().max().item()
    print("\n[Claim 2] batched replay determinism (ratio=1 anchor):", flush=True)
    print(f"  replay#1 vs #2: max|ratio-1| = {max_ratio_dev:.3e} (expect 0)", flush=True)
    ok2 = max_ratio_dev < 1e-6

    # ---- Claim 3: grad flows through batched replay ----
    print("\n[Claim 3] backward through batched replay:", flush=True)
    stage_b.model.transformer.train()
    try:
        rep = stage_b.replay(conds, segment=seg, params=params)
        adv = torch.randn(B, device="cuda")
        loss = (adv.unsqueeze(1) * rep.log_probs).mean()
        loss.backward()
        gnorm = (
            sum(p.grad.float().norm().item() ** 2 for p in stage_b.model.transformer.parameters() if p.grad is not None)
            ** 0.5
        )
        ok3 = gnorm > 0
        print(f"  loss={loss.item():.4f}  grad_norm={gnorm:.4e}  -> {'OK' if ok3 else 'FAIL (no grad)'}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok3 = False
        print(f"  FAILED: {e}", flush=True)
    stage_b.model.transformer.zero_grad(set_to_none=True)
    stage_b.model.transformer.eval()

    print("\n==== RESULT ====", flush=True)
    print(f"  Claim 1 (batched==serial, bf16 tol + mapping): {'PASS' if ok1 else 'FAIL'}", flush=True)
    print(f"  Claim 2 (batched replay deterministic ratio=1): {'PASS' if ok2 else 'FAIL'}", flush=True)
    print(f"  Claim 3 (grad flows through batched replay):    {'PASS' if ok3 else 'FAIL'}", flush=True)

    if os.environ.get("REPLAY_BENCH", "1") == "1":
        _bench(stage, stage_b, conds, seg, params)


def _bench(stage, stage_b, conds, seg, params) -> None:
    print("\n[bench] serial vs batched replay at PE micro geometry (S={} steps)".format(len(SDE_INDICES)), flush=True)

    def timed(fn, iters=20, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    with torch.no_grad():
        ts = timed(lambda: stage.replay(conds, segment=seg, params=params))
        tb = timed(lambda: stage_b.replay(conds, segment=seg, params=params))
    print(
        f"  forward (no_grad):  serial={ts * 1e3:.2f}ms  batched={tb * 1e3:.2f}ms  speedup={ts / tb:.2f}x", flush=True
    )

    def fwd_bwd(stg):
        stg.model.transformer.train()
        stg.model.transformer.zero_grad(set_to_none=True)
        rep = stg.replay(conds, segment=seg, params=params)
        (rep.log_probs.mean()).backward()
        stg.model.transformer.eval()

    tsg = timed(lambda: fwd_bwd(stage), iters=12, warmup=3)
    tbg = timed(lambda: fwd_bwd(stage_b), iters=12, warmup=3)
    print(
        f"  fwd+bwd:            serial={tsg * 1e3:.2f}ms  batched={tbg * 1e3:.2f}ms  speedup={tsg / tbg:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
