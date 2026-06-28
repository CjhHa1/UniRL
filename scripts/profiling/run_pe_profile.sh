#!/usr/bin/env bash
# Profile the PE trainside pipeline (Qwen3 rewrite + SD3 diffusion + PickScore).
set -euo pipefail

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
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export UNIRL_PROFILE_PE_TIMING="${UNIRL_PROFILE_PE_TIMING:-1}"

LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-/data/models}"
STAGE_MODELS="${STAGE_MODELS:-1}"
SD3_SOURCE="${SD3_SOURCE:-/apdcephfs_hldy/share_305110755/hunyuan/public_models/stabilityai/stable-diffusion-3.5-medium}"
MODELS_SOURCE_DIR="${MODELS_SOURCE_DIR:-/apdcephfs/private_aimicahchen/models}"
SD3_MODEL="${SD3_MODEL:-${SD3_SOURCE}}"
LLM_MODEL="${LLM_MODEL:-${MODELS_SOURCE_DIR}/Qwen3-0.6B}"
PICKSCORE_PROCESSOR_ID="${PICKSCORE_PROCESSOR_ID:-${MODELS_SOURCE_DIR}/CLIP-ViT-H-14-laion2B}"
PICKSCORE_MODEL_ID="${PICKSCORE_MODEL_ID:-${MODELS_SOURCE_DIR}/PickScore_v1}"

NUM_DEVICES="${NUM_DEVICES:-8}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
PE_BATCH_SIZE="${PE_BATCH_SIZE:-8}"
PROFILE_OUT="${PROFILE_OUT:-outputs/profiling/pe_sd3_qwen3_$(date '+%Y%m%d_%H%M%S')}"
START_RAY="${START_RAY:-1}"
STOP_RAY_ON_EXIT="${STOP_RAY_ON_EXIT:-${START_RAY}}"

mkdir -p "${PROFILE_OUT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

stage_model_dir() {
    local src="$1"
    local dst="$2"
    local marker="${dst}/.unirl_stage_complete"
    if [[ ! -d "${src}" ]]; then
        echo "Missing model source: ${src}" >&2
        return 1
    fi
    if [[ -f "${marker}" ]]; then
        log "Using staged model: ${dst}"
        return 0
    fi
    mkdir -p "${dst}"
    log "Staging model: ${src} -> ${dst}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete --partial --info=progress2 "${src}/" "${dst}/"
    else
        cp -a "${src}/." "${dst}/"
    fi
    touch "${marker}"
    log "Staged model: ${dst}"
}

start_mem_sampler() {
    local outfile="$1"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
            --format=csv -l 2 > "${outfile}" &
        echo $!
    fi
}

stop_mem_sampler() {
    local pid="${1:-}"
    if [[ -n "${pid}" ]]; then
        kill "${pid}" 2>/dev/null || true
    fi
}

mem_pid=""
cleanup() {
    stop_mem_sampler "${mem_pid}"
    if [[ "${STOP_RAY_ON_EXIT}" == "1" ]]; then
        ray stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${STAGE_MODELS}" == "1" ]]; then
    mkdir -p "${LOCAL_MODELS_DIR}"
    stage_model_dir "${SD3_SOURCE}" "${LOCAL_MODELS_DIR}/stable-diffusion-3.5-medium"
    stage_model_dir "${MODELS_SOURCE_DIR}/Qwen3-0.6B" "${LOCAL_MODELS_DIR}/Qwen3-0.6B"
    stage_model_dir "${MODELS_SOURCE_DIR}/CLIP-ViT-H-14-laion2B" "${LOCAL_MODELS_DIR}/CLIP-ViT-H-14-laion2B"
    stage_model_dir "${MODELS_SOURCE_DIR}/PickScore_v1" "${LOCAL_MODELS_DIR}/PickScore_v1"
    SD3_MODEL="${LOCAL_MODELS_DIR}/stable-diffusion-3.5-medium"
    LLM_MODEL="${LOCAL_MODELS_DIR}/Qwen3-0.6B"
    PICKSCORE_PROCESSOR_ID="${LOCAL_MODELS_DIR}/CLIP-ViT-H-14-laion2B"
    PICKSCORE_MODEL_ID="${LOCAL_MODELS_DIR}/PickScore_v1"
fi

if [[ "${START_RAY}" == "1" ]]; then
    ray stop 2>/dev/null || true
    sleep 3
    ray start --head --node-ip-address=127.0.0.1 --port="${RAY_PORT:-6379}" \
        --dashboard-host=0.0.0.0 --num-gpus="${NUM_DEVICES}"
    sleep 5
fi

log "PE profiling start"
log "repo=${REPO_ROOT}"
log "sd3=${SD3_MODEL}"
log "llm=${LLM_MODEL}"
log "out=${PROFILE_OUT}"

mem_pid="$(start_mem_sampler "${PROFILE_OUT}/gpu_memory.csv" || true)"

set +e
PRETRAINED_MODEL="${SD3_MODEL}" \
LLM_MODEL="${LLM_MODEL}" \
PICKSCORE_PROCESSOR_ID="${PICKSCORE_PROCESSOR_ID}" \
PICKSCORE_MODEL_ID="${PICKSCORE_MODEL_ID}" \
python -m unirl.train_pe \
    --config-name=pe/pe_trainside_pickscore \
    num_devices="${NUM_DEVICES}" \
    num_rollouts="${NUM_ROLLOUTS}" \
    batch_size="${PE_BATCH_SIZE}" \
    logging.report_to_wandb=false \
    reward.backend.config.processor_id="${PICKSCORE_PROCESSOR_ID}" \
    reward.backend.config.model_id="${PICKSCORE_MODEL_ID}" \
    "$@" \
    2>&1 | tee "${PROFILE_OUT}/run.log"
train_status=${PIPESTATUS[0]}
set -e

stop_mem_sampler "${mem_pid}"
mem_pid=""

python scripts/profiling/analyze_bottlenecks.py \
    --profile pe \
    --config examples/pe/pe_trainside_pickscore.yaml \
    --log "${PROFILE_OUT}/run.log" \
    --memory-csv "${PROFILE_OUT}/gpu_memory.csv" \
    --output-json "${PROFILE_OUT}/bottlenecks.json" \
    --markdown "${PROFILE_OUT}/bottlenecks.md"

if [[ "${train_status}" -ne 0 ]]; then
    log "PE profiling failed with exit code ${train_status}: ${PROFILE_OUT}"
    exit "${train_status}"
fi

log "PE profiling done: ${PROFILE_OUT}"
