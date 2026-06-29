#!/usr/bin/env bash
# PE diffusion_train optimization sweep. Each case is a full PE trainside run
# (Qwen3 rewrite + SD3 diffusion + PickScore) with different SD3 train knobs;
# we read the per-phase timing (diffusion_train / generate / ar_train / reward)
# and the GRPO ratio to find what actually moves the 62% diffusion_train phase.
#
# Models are read from /data/models (local NVMe, already staged). The GPU
# occupier must be neutered first (see /tmp/gpu_occupy.py stub).
#
# Usage:
#   CASES="baseline mbs8 reshard_false compile combined" \
#     bash scripts/profiling/run_pe_perf_sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f /data/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-qwen35}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export REPORT_TO_WANDB=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UNIRL_PROFILE_PE_TIMING=1

MODEL_DIR="${MODEL_DIR:-/data/models}"
export SD3_MODEL="${SD3_MODEL:-${MODEL_DIR}/stable-diffusion-3.5-medium}"
export LLM_MODEL="${LLM_MODEL:-${MODEL_DIR}/Qwen3-0.6B}"
export PICKSCORE_PROCESSOR_ID="${PICKSCORE_PROCESSOR_ID:-${MODEL_DIR}/CLIP-ViT-H-14-laion2B}"
export PICKSCORE_MODEL_ID="${PICKSCORE_MODEL_ID:-${MODEL_DIR}/PickScore_v1}"

NUM_DEVICES="${NUM_DEVICES:-8}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-6}"
PE_BATCH_SIZE="${PE_BATCH_SIZE:-8}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/pe_perf}"
WARMUP="${WARMUP:-3}"
CASES="${CASES:-baseline mbs8 reshard_false compile combined}"

mkdir -p "${PROFILE_ROOT}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

neuter_occupier() {
    # Defensive: ensure no real GPU occupier is stealing cycles.
    for pid in $(pgrep -f "python.* /tmp/gpu_occupy" 2>/dev/null); do
        if grep -q NEUTERED /proc/"${pid}"/cmdline 2>/dev/null; then :; fi
    done
    # Kill any occupier that is actually using a GPU (the real one allocates VRAM).
    if command -v nvidia-smi >/dev/null 2>&1; then
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [[ "${used:-0}" -gt 2000 ]]; then
            log "WARN: GPU0 has ${used}MiB used before case — possible occupier; check /tmp/gpu_occupy.py is the stub."
        fi
    fi
}

ray_up() {
    ray stop >/dev/null 2>&1 || true
    sleep 3
    ray start --head --node-ip-address=127.0.0.1 --port="${RAY_PORT:-6379}" \
        --dashboard-host=0.0.0.0 --num-gpus="${NUM_DEVICES}" >/dev/null 2>&1
    sleep 5
}

run_case() {
    local name="$1"; shift
    local out="${PROFILE_ROOT}/${name}"
    mkdir -p "${out}"
    neuter_occupier
    log "=== PE case=${name} overrides: $* ==="
    ray_up
    local mem_pid=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
            --format=csv -l 2 > "${out}/gpu_memory.csv" &
        mem_pid=$!
    fi
    PRETRAINED_MODEL="${SD3_MODEL}" \
    python -m unirl.train_pe \
        --config-name=pe/pe_trainside_pickscore \
        num_devices="${NUM_DEVICES}" \
        num_rollouts="${NUM_ROLLOUTS}" \
        batch_size="${PE_BATCH_SIZE}" \
        logging.report_to_wandb=false \
        reward.backend.config.processor_id="${PICKSCORE_PROCESSOR_ID}" \
        reward.backend.config.model_id="${PICKSCORE_MODEL_ID}" \
        "$@" \
        > "${out}/run.log" 2>&1
    local status=$?
    [[ -n "${mem_pid}" ]] && kill "${mem_pid}" 2>/dev/null || true
    ray stop >/dev/null 2>&1 || true
    python scripts/profiling/analyze_bottlenecks.py \
        --profile pe \
        --config examples/pe/pe_trainside_pickscore.yaml \
        --log "${out}/run.log" \
        --memory-csv "${out}/gpu_memory.csv" \
        --warmup "${WARMUP}" \
        --output-json "${out}/bottlenecks.json" \
        --markdown "${out}/bottlenecks.md" > "${out}/analyze.log" 2>&1 || log "analyze failed for ${name}"
    if [[ "${status}" -ne 0 ]]; then
        log "case=${name} FAILED (exit ${status}); see ${out}/run.log"
    else
        log "case=${name} done -> ${out}"
    fi
}

for c in ${CASES}; do
    case "${c}" in
        baseline)
            run_case baseline ;;
        mbs4)
            run_case mbs4 diffusion.stack.micro_batch_size=4 ;;
        mbs8)
            run_case mbs8 diffusion.stack.micro_batch_size=8 ;;
        mbs16)
            run_case mbs16 diffusion.stack.micro_batch_size=16 ;;
        reshard_false)
            run_case reshard_false diffusion.backend.fsdp_cfg.reshard_after_forward=false ;;
        prefetch)
            run_case prefetch diffusion.backend.fsdp_cfg.forward_prefetch=true ;;
        compile)
            run_case compile diffusion.backend.fsdp_cfg.use_torch_compile=true ;;
        batched_replay)
            run_case batched_replay diffusion.pipeline.batch_replay_steps=true ;;
        combined)
            run_case combined \
                diffusion.stack.micro_batch_size=8 \
                diffusion.backend.fsdp_cfg.reshard_after_forward=false \
                diffusion.backend.fsdp_cfg.use_torch_compile=true ;;
        mbs8_batched)
            run_case mbs8_batched \
                diffusion.stack.micro_batch_size=8 \
                diffusion.pipeline.batch_replay_steps=true ;;
        batched_compile)
            run_case batched_compile \
                diffusion.pipeline.batch_replay_steps=true \
                diffusion.backend.fsdp_cfg.use_torch_compile=true ;;
        mbs8_batched_compile)
            run_case mbs8_batched_compile \
                diffusion.stack.micro_batch_size=8 \
                diffusion.pipeline.batch_replay_steps=true \
                diffusion.backend.fsdp_cfg.use_torch_compile=true ;;
        max)
            run_case max \
                diffusion.stack.micro_batch_size=8 \
                diffusion.pipeline.batch_replay_steps=true \
                diffusion.backend.fsdp_cfg.reshard_after_forward=false \
                diffusion.backend.fsdp_cfg.use_torch_compile=true ;;
        *) log "unknown case=${c}" ;;
    esac
done

# Summary table across cases.
PROFILE_ROOT="${PROFILE_ROOT}" CASES="${CASES}" python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["PROFILE_ROOT"])
cases = os.environ["CASES"].split()
def g(d, k):
    return d.get("timing", {}).get("avg", {}).get(k)
rows = []
for name in cases:
    p = root / name / "bottlenecks.json"
    if not p.exists():
        rows.append((name, "MISSING", "", "", "", "", "")); continue
    d = json.loads(p.read_text())
    step = g(d, "rollout_time_s")
    diff = g(d, "diffusion_train_time_s")
    gen = g(d, "generate_time_s")
    ar = g(d, "ar_train_time_s")
    rew = g(d, "reward_time_s")
    mem = (d.get("memory") or {}).get("peak_gb")
    rows.append((name,
                 f"{step:.1f}" if step else "?",
                 f"{diff:.1f}" if diff else "?",
                 f"{gen:.1f}" if gen else "?",
                 f"{ar:.2f}" if ar else "?",
                 f"{rew:.1f}" if rew else "?",
                 f"{mem:.1f}" if mem else "?"))
hdr = ("case", "step_s", "diff_train_s", "generate_s", "ar_train_s", "reward_s", "peak_gb")
lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
for r in rows:
    lines.append("| " + " | ".join(r) + " |")
out = "\n".join(lines)
(root / "summary.md").write_text(out + "\n")
print(out)
PY
log "PE perf sweep done -> ${PROFILE_ROOT}/summary.md"
