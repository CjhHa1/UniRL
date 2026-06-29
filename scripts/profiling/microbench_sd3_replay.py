#!/usr/bin/env python
"""SD3.5-medium training-forward throughput microbench (PE diffusion bottleneck).

PE's 62%% bottleneck is the SD3 *train* phase (``diffusion.stack.train_track``).
That phase replays the rollout's SDE transitions through the SD3 transformer at
``micro_batch_size=1`` and ONE SDE step at a time -- i.e. a long stream of tiny
batch=1 transformer forwards/backwards. SD3.5-medium is small (24 layers,
hidden 1536, ~1037 image + 333 text tokens at 512px), so the hypothesis is that
batch=1 is heavily *launch-bound* (the opposite of compute-bound BAGEL-7B).

This script isolates the SD3 transformer (no text encoders / VAE / Ray / FSDP)
and measures, on ONE GPU:

  A. forward throughput vs batch size B (no_grad)         -> underutilization @ B=1
  B. forward+backward throughput vs batch size B          -> train-step scaling
  C. serial-S-steps vs one batched-(S*B) forward          -> motivates batched replay

The decisive numbers: samples/s at B=1 vs the saturation batch, and the implied
speedup from batching the train forward (micro_batch_size>1 and/or batched-step
replay). Run with the occupier neutered:

  CUDA_VISIBLE_DEVICES=0 SD3_DIR=/data/models/stable-diffusion-3.5-medium \
    python scripts/profiling/microbench_sd3_replay.py
"""

from __future__ import annotations

import os
import time

import torch

SD3_DIR = os.environ.get("SD3_DIR", "/data/models/stable-diffusion-3.5-medium")
# 512x512 SD3 latent: 16 x 64 x 64; text embeds 333 x 4096; pooled 2048.
H = W = int(os.environ.get("LATENT_HW", "64"))
SEQ_TXT = int(os.environ.get("SEQ_TXT", "333"))
ITERS = int(os.environ.get("ITERS", "12"))
WARMUP = int(os.environ.get("WARMUP", "4"))
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,4,8,16,24,32,48,64").split(",")]
DTYPE = torch.bfloat16


def _load_transformer():
    from diffusers import SD3Transformer2DModel

    print(f"[microbench] loading SD3 transformer from {SD3_DIR}/transformer ...", flush=True)
    tf = SD3Transformer2DModel.from_pretrained(f"{SD3_DIR}/transformer", torch_dtype=DTYPE)
    tf = tf.to("cuda").eval()
    n_params = sum(p.numel() for p in tf.parameters())
    print(f"[microbench] transformer params: {n_params / 1e9:.2f}B; layers={tf.config.num_layers}", flush=True)
    return tf


def _inputs(b: int):
    x = torch.randn(b, 16, H, W, device="cuda", dtype=DTYPE)
    enc = torch.randn(b, SEQ_TXT, 4096, device="cuda", dtype=DTYPE)
    pooled = torch.randn(b, 2048, device="cuda", dtype=DTYPE)
    t = torch.rand(b, device="cuda", dtype=DTYPE) * 1000.0
    return x, enc, pooled, t


def _fwd(tf, x, enc, pooled, t):
    return tf(hidden_states=x, encoder_hidden_states=enc, timestep=t, pooled_projections=pooled, return_dict=False)[0]


def _time(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_forward(tf):
    print("\n==== A. forward throughput vs batch (no_grad) ====", flush=True)
    print(f"  {'B':>4} | {'ms/fwd':>9} | {'ms/sample':>10} | {'samples/s':>10} | {'vs B=1':>8}", flush=True)
    base_per_sample = None
    rows = []
    with torch.no_grad():
        for b in BATCHES:
            x, enc, pooled, t = _inputs(b)
            try:
                dt = _time(lambda: _fwd(tf, x, enc, pooled, t))
            except RuntimeError as e:
                print(f"  {b:>4} | OOM/err: {str(e)[:50]}", flush=True)
                continue
            per_sample = dt / b
            if base_per_sample is None:
                base_per_sample = per_sample
            sps = b / dt
            speedup = base_per_sample / per_sample  # >1 means per-sample got cheaper
            rows.append((b, dt, per_sample, sps, speedup))
            print(
                f"  {b:>4} | {dt * 1e3:>9.2f} | {per_sample * 1e3:>10.3f} | {sps:>10.1f} | {speedup:>7.2f}x",
                flush=True,
            )
            torch.cuda.empty_cache()
    if rows:
        best = max(rows, key=lambda r: r[3])
        print(
            f"\n  -> B=1: {rows[0][3]:.1f} samples/s; peak {best[3]:.1f} samples/s @ B={best[0]} "
            f"({best[3] / rows[0][3]:.1f}x throughput). If >>1, B=1 is launch-bound (underutilized).",
            flush=True,
        )
    return rows


def bench_fwd_bwd(tf):
    print("\n==== B. forward+backward throughput vs batch (full-param grad; LoRA would be lighter) ====", flush=True)
    print(f"  {'B':>4} | {'ms/step':>9} | {'ms/sample':>10} | {'samples/s':>10} | {'vs B=1':>8}", flush=True)
    tf.train()
    base_per_sample = None
    rows = []

    def step(x, enc, pooled, t):
        tf.zero_grad(set_to_none=True)
        out = _fwd(tf, x, enc, pooled, t)
        # Stand-in for the GRPO log-prob loss: a scalar that exercises the full
        # backward graph (advantage-weighted log-prob is also a scalar over the latent).
        loss = out.float().pow(2).mean()
        loss.backward()

    for b in BATCHES:
        x, enc, pooled, t = _inputs(b)
        try:
            dt = _time(lambda: step(x, enc, pooled, t), iters=max(6, ITERS // 2), warmup=3)
        except RuntimeError as e:
            print(f"  {b:>4} | OOM/err: {str(e)[:50]}", flush=True)
            torch.cuda.empty_cache()
            continue
        per_sample = dt / b
        if base_per_sample is None:
            base_per_sample = per_sample
        rows.append((b, dt, per_sample, b / dt, base_per_sample / per_sample))
        print(
            f"  {b:>4} | {dt * 1e3:>9.2f} | {per_sample * 1e3:>10.3f} | {b / dt:>10.1f} | "
            f"{base_per_sample / per_sample:>7.2f}x",
            flush=True,
        )
        torch.cuda.empty_cache()
    tf.eval()
    if rows:
        best = max(rows, key=lambda r: r[3])
        print(
            f"\n  -> fwd+bwd: B=1 {rows[0][3]:.1f} samples/s; peak {best[3]:.1f} @ B={best[0]} "
            f"({best[3] / rows[0][3]:.1f}x).",
            flush=True,
        )
    return rows


def bench_serial_vs_batched_steps(tf):
    """C. Replay does S SDE steps. Today each step is a separate forward at the
    micro batch B. Batched-step replay stacks the S steps into ONE forward of
    S*B. Compare S serial forwards (bs=B) vs 1 forward (bs=S*B)."""
    print("\n==== C. serial S-step forwards vs one batched (S*B) forward (no_grad) ====", flush=True)
    S = int(os.environ.get("SDE_STEPS", "3"))
    print(f"  S={S} SDE steps per sample", flush=True)
    print(f"  {'B':>4} | {'serial S x bs=B (ms)':>20} | {'batched bs=S*B (ms)':>20} | {'speedup':>8}", flush=True)
    with torch.no_grad():
        for b in (1, 4, 8, 16):
            xb, encb, pb, tb = _inputs(b)
            xsb, encsb, psb, tsb = _inputs(S * b)

            def serial():
                for _ in range(S):
                    _fwd(tf, xb, encb, pb, tb)

            def batched():
                _fwd(tf, xsb, encsb, psb, tsb)

            try:
                ts = _time(serial)
                tbat = _time(batched)
            except RuntimeError as e:
                print(f"  {b:>4} | err: {str(e)[:40]}", flush=True)
                continue
            print(f"  {b:>4} | {ts * 1e3:>20.2f} | {tbat * 1e3:>20.2f} | {ts / tbat:>7.2f}x", flush=True)
            torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    print(f"[microbench] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tf = _load_transformer()
    bench_forward(tf)
    bench_fwd_bwd(tf)
    bench_serial_vs_batched_steps(tf)
    print("\n[microbench] done.", flush=True)


if __name__ == "__main__":
    main()
