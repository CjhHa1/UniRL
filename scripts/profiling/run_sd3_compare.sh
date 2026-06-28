#!/usr/bin/env bash
# Compare SD3 native FSDP, native optimized knobs, and VeOmni backend.
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

MODEL_DIR="${MODEL_DIR:-/data/models}"
SD3_MODEL="${SD3_MODEL:-${MODEL_DIR}/stable-diffusion-3.5-medium}"
PICKSCORE_PROCESSOR_ID="${PICKSCORE_PROCESSOR_ID:-${MODEL_DIR}/CLIP-ViT-H-14-laion2B}"
PICKSCORE_MODEL_ID="${PICKSCORE_MODEL_ID:-${MODEL_DIR}/PickScore_v1}"
NUM_DEVICES="${NUM_DEVICES:-8}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-3}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/sd3_compare_$(date '+%Y%m%d_%H%M%S')}"
START_RAY="${START_RAY:-1}"
STOP_RAY_ON_EXIT="${STOP_RAY_ON_EXIT:-${START_RAY}}"

mkdir -p "${PROFILE_ROOT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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

cleanup() {
    stop_mem_sampler "${mem_pid:-}"
    if [[ "${STOP_RAY_ON_EXIT}" == "1" ]]; then
        ray stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_ray() {
    if [[ "${START_RAY}" == "1" ]]; then
        ray stop 2>/dev/null || true
        sleep 3
        ray start --head --node-ip-address=127.0.0.1 --port="${RAY_PORT:-6379}" \
            --dashboard-host=0.0.0.0 --num-gpus="${NUM_DEVICES}"
        sleep 5
    fi
}

run_case() {
    local name="$1"
    local config_name="$2"
    shift 2
    local out="${PROFILE_ROOT}/${name}"
    mkdir -p "${out}"
    log "SD3 compare case=${name} config=${config_name}"
    start_ray
    mem_pid="$(start_mem_sampler "${out}/gpu_memory.csv" || true)"
    set +e
    PRETRAINED_MODEL="${SD3_MODEL}" \
    PICKSCORE_PROCESSOR_ID="${PICKSCORE_PROCESSOR_ID}" \
    PICKSCORE_MODEL_ID="${PICKSCORE_MODEL_ID}" \
    python -m unirl.train_diffusion \
        --config-name="${config_name}" \
        num_devices="${NUM_DEVICES}" \
        +num_rollouts="${NUM_ROLLOUTS}" \
        logging.report_to_wandb=false \
        reward.backend.config.processor_id="${PICKSCORE_PROCESSOR_ID}" \
        reward.backend.config.model_id="${PICKSCORE_MODEL_ID}" \
        "$@" \
        2>&1 | tee "${out}/run.log"
    local status=${PIPESTATUS[0]}
    set -e
    stop_mem_sampler "${mem_pid}"
    mem_pid=""
    python scripts/profiling/analyze_bottlenecks.py \
        --profile sd3 \
        --config "examples/${config_name}.yaml" \
        --log "${out}/run.log" \
        --memory-csv "${out}/gpu_memory.csv" \
        --output-json "${out}/bottlenecks.json" \
        --markdown "${out}/bottlenecks.md" || true
    if [[ "${START_RAY}" == "1" ]]; then
        ray stop 2>/dev/null || true
    fi
    if [[ "${status}" -ne 0 ]]; then
        log "case=${name} failed with exit code ${status}"
        return "${status}"
    fi
}

# Optional global batch override (keeps the compare quick; relative deltas are
# what matter for the prefetch/compile levers). Empty = recipe default (48).
SD3_BATCH="${SD3_BATCH:-16}"
batch_override=()
[[ -n "${SD3_BATCH}" ]] && batch_override=(batch_size="${SD3_BATCH}")

CASES="${CASES:-native native_compile_prefetch veomni}"
# Drop the veomni case when its backend isn't importable in this env.
if echo " ${CASES} " | grep -q " veomni " && ! python -c "import veomni" 2>/dev/null; then
    log "veomni backend not importable in this env; dropping the veomni case"
    CASES="$(echo "${CASES}" | sed 's/\bveomni\b//g')"
fi

for c in ${CASES}; do
    case "${c}" in
        native)
            run_case native diffusion/sd3/sd3_trainside "${batch_override[@]}" || log "native case failed"
            ;;
        native_compile_prefetch)
            run_case native_compile_prefetch diffusion/sd3/sd3_trainside "${batch_override[@]}" \
                backend.fsdp_cfg.forward_prefetch=true \
                backend.fsdp_cfg.use_torch_compile=true || log "native_compile_prefetch case failed"
            ;;
        veomni)
            run_case veomni diffusion/sd3_trainside_veomni "${batch_override[@]}" || log "veomni case failed"
            ;;
        *) log "unknown case=${c}" ;;
    esac
done

PROFILE_ROOT="${PROFILE_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PROFILE_ROOT"])
rows = []
for name in ("native", "native_compile_prefetch", "veomni"):
    path = root / name / "bottlenecks.json"
    if not path.exists():
        rows.append((name, "missing", "", "", "", ""))
        continue
    data = json.loads(path.read_text())
    avg = data.get("timing", {}).get("avg", {})
    mem = data.get("memory") or {}
    rows.append(
        (
            name,
            f"{avg.get('rollout_time_s', 0):.2f}",
            f"{avg.get('generate_time_s', 0):.2f}",
            f"{avg.get('train_time_s', 0):.2f}",
            f"{avg.get('reward_time_s', 0):.2f}",
            f"{mem.get('peak_gb', 0):.1f}",
        )
    )

lines = [
    "# SD3 Native vs VeOmni Profiling",
    "",
    "| case | step_s | generate_s | train_s | reward_s | peak_gb |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append("| " + " | ".join(row) + " |")
(root / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

log "SD3 comparison done: ${PROFILE_ROOT}"
