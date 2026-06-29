#!/usr/bin/env python
"""Decompose the PE `generate` phase (now the #1 phase after the diff_train win).

generate = Qwen3 rewrite (AR decode) + SD3 text-embed + SD3 denoise + SD3 VAE
decode, run sequentially per DP worker. PE geometry per GPU (P=8,N=4,M=8 / 8 GPU):
  - 4 Qwen3 rewrites  (max_new_tokens=512)
  - 32 SD3 images = text-embed(32 texts, 4 unique) + denoise(32, 10 steps) + VAE(32)

This times each component on ONE GPU (stages called directly) so we know where
generate's ~15s goes and what is worth optimizing next. Occupier must be neutered.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python scripts/profiling/microbench_pe_generate.py
"""

from __future__ import annotations

import os
import time

import torch

from unirl.models.qwen3.config import Qwen3PipelineConfig
from unirl.models.qwen3.pipeline import Qwen3Pipeline
from unirl.models.sd3.config import SD3PipelineConfig
from unirl.models.sd3.pipeline import SD3Pipeline
from unirl.models.types.ar import ARSamplingParams
from unirl.sde.kernels import FlowSDEStrategy
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.primitives import Texts
from unirl.types.sampling import DiffusionSamplingParams

SD3_DIR = os.environ.get("SD3_DIR", "/data/models/stable-diffusion-3.5-medium")
LLM_DIR = os.environ.get("LLM_DIR", "/data/models/Qwen3-0.6B")
N_AR = int(os.environ.get("N_AR", "4"))  # P*N / 8 GPU = 4 rewrites / GPU
N_SD3 = int(os.environ.get("N_SD3", "32"))  # P*N*M / 8 GPU = 32 images / GPU
N_UNIQUE = int(os.environ.get("N_UNIQUE", "4"))  # 4 unique rewrites -> M=8 each
T = int(os.environ.get("T", "10"))
MAXNEW = int(os.environ.get("MAXNEW", "512"))


def _timed(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    torch.manual_seed(0)
    print(
        f"[pe-gen] AR rewrites/GPU={N_AR} (max_new={MAXNEW}); SD3 images/GPU={N_SD3} "
        f"({N_UNIQUE} unique prompts), T={T} steps",
        flush=True,
    )

    # ---- Qwen3 rewrite (AR decode) ----
    qpipe = Qwen3Pipeline.from_config(Qwen3PipelineConfig(pretrained_model_ckpt_path=LLM_DIR, model_precision="bf16"))
    qpipe.ar.model.transformer.eval()
    prompts = [f"a photo of scene number {i}, highly detailed" for i in range(N_AR)]
    conds = qpipe.chat_template.embed(Texts(texts=prompts))
    sp = ARSamplingParams(max_new_tokens=MAXNEW, temperature=0.7, top_p=0.9, top_k=0, stop_token_id=None)
    from unirl.models.qwen3.ar import Qwen3ARParams

    qp = Qwen3ARParams(max_tokens=MAXNEW, temperature=0.7, top_p=0.9, top_k=0)

    def _ar():
        with torch.no_grad():
            qpipe.ar.autoregress(conds, sampling_params=sp, params=qp)

    t_ar = _timed(_ar, iters=3, warmup=1)
    print(f"  qwen3_rewrite (AR decode, {N_AR}x{MAXNEW}tok): {t_ar * 1e3:8.1f} ms", flush=True)
    torch.cuda.empty_cache()  # Qwen3 (~1.2GB) stays resident alongside SD3; both fit on one H20

    # ---- SD3 stages ----
    spipe = SD3Pipeline.from_config(
        SD3PipelineConfig(pretrained_model_ckpt_path=SD3_DIR, model_precision="bf16", shift=3.0),
        strategy=FlowSDEStrategy(),
    )
    spipe.diffusion.model.transformer.eval()
    rewrites = [f"a stunning rendering of unique prompt {i % N_UNIQUE}, cinematic lighting" for i in range(N_SD3)]
    texts = Texts(texts=rewrites)
    schedule = get_sigma_schedule(T, shift=3.0, device=torch.device("cuda"))
    params = DiffusionSamplingParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=512,
        width=512,
        eta=0.7,
        seed=0,
        sde_indices=[0, 2, 4],
    )

    def _embed():
        with torch.no_grad():
            spipe.text_embed.embed(texts)

    t_embed = _timed(_embed, iters=3, warmup=1)
    cond = spipe.text_embed.embed(texts)
    from unirl.models.sd3.conditions import SD3Conditions

    sd3_cond = SD3Conditions(text=cond, negative_text=None)

    seg_holder = {}

    def _denoise():
        with torch.no_grad():
            seg_holder["seg"] = spipe.diffusion.diffuse(sd3_cond, schedule=schedule, params=params)

    t_denoise = _timed(_denoise, iters=3, warmup=1)
    seg = seg_holder["seg"]

    def _vae():
        with torch.no_grad():
            spipe.vae_decode.decode(seg)

    t_vae = _timed(_vae, iters=2, warmup=1)

    print(f"  sd3_text_embed ({N_SD3} texts, {N_UNIQUE} uniq): {t_embed * 1e3:8.1f} ms", flush=True)
    print(f"  sd3_denoise ({N_SD3}x{T} steps):               {t_denoise * 1e3:8.1f} ms", flush=True)
    print(f"  sd3_vae_decode ({N_SD3} latents):              {t_vae * 1e3:8.1f} ms", flush=True)

    total = t_ar + t_embed + t_denoise + t_vae
    print("\n  ---- per-GPU generate decomposition ----", flush=True)
    for name, t in [
        ("qwen3_rewrite", t_ar),
        ("sd3_text_embed", t_embed),
        ("sd3_denoise", t_denoise),
        ("sd3_vae_decode", t_vae),
    ]:
        print(f"    {name:<16} {t * 1e3:8.1f} ms  ({t / total * 100:4.1f}%)", flush=True)
    print(f"    {'TOTAL':<16} {total * 1e3:8.1f} ms", flush=True)


if __name__ == "__main__":
    main()
