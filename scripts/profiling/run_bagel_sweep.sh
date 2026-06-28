#!/usr/bin/env bash
# BAGEL-7B-MoT diffusion optimization sweep: isolate each profiling-report lever
# against the same recipe so the deltas are directly comparable.
#
#   baseline   steps=14, SDE=2, context-cache ON  (current default)
#   nocache    context-cache OFF                   (= the report's original 183.1s baseline)
#   steps10    num_inference_steps 14 -> 10        (report P0: ~30% rollout)
#   sde1       num_sde_steps 2 -> 1                (report P0: fewer SDE forwards)
#
# Each case runs from a cold ray head + fresh model load so memory is isolated.
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
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_DIR="${MODEL_DIR:-/data/models}"
export BAGEL_PATH="${BAGEL_PATH:-${MODEL_DIR}/BAGEL-7B-MoT}"
export PICKSCORE_PROCESSOR_ID="${PICKSCORE_PROCESSOR_ID:-${MODEL_DIR}/CLIP-ViT-H-14-laion2B}"
export PICKSCORE_MODEL_ID="${PICKSCORE_MODEL_ID:-${MODEL_DIR}/PickScore_v1}"
NUM_DEVICES="${NUM_DEVICES:-8}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"
WARMUP="${WARMUP:-1}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/bagel_sweep_$(date '+%Y%m%d_%H%M%S')}"
START_RAY="${START_RAY:-1}"
# Space-separated case names to run (default: all four).
CASES="${CASES:-baseline nocache steps10 sde1}"

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
stop_mem_sampler() { [[ -n "${1:-}" ]] && kill "$1" 2>/dev/null || true; }

mem_pid=""
cleanup() {
    stop_mem_sampler "${mem_pid:-}"
    ray stop 2>/dev/null || true
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

case_overrides() {
    case "$1" in
        baseline) echo "" ;;
        nocache)  echo "++pipeline.cache_t2i_contexts=false" ;;
        steps10)  echo "sampling.num_inference_steps=10" ;;
        sde1)     echo "sampling.scheduler.num_sde_steps=1" ;;
        *) echo "__UNKNOWN__" ;;
    esac
}

run_case() {
    local name="$1"
    local overrides
    overrides="$(case_overrides "${name}")"
    if [[ "${overrides}" == "__UNKNOWN__" ]]; then
        log "skip unknown case=${name}"; return 0
    fi
    local out="${PROFILE_ROOT}/${name}"
    mkdir -p "${out}"
    log "BAGEL sweep case=${name} overrides='${overrides}'"
    start_ray
    mem_pid="$(start_mem_sampler "${out}/gpu_memory.csv" || true)"
    set +e
    # shellcheck disable=SC2086
    python -m unirl.train_diffusion \
        --config-name=diffusion/bagel/bagel_trainside_lora \
        num_devices="${NUM_DEVICES}" \
        num_rollouts="${NUM_ROLLOUTS}" \
        logging.report_to_wandb=false \
        ${overrides} \
        2>&1 | tee "${out}/run.log"
    local status=${PIPESTATUS[0]}
    set -e
    stop_mem_sampler "${mem_pid}"; mem_pid=""
    python scripts/profiling/analyze_bottlenecks.py \
        --profile bagel \
        --config examples/diffusion/bagel/bagel_trainside_lora.yaml \
        --log "${out}/run.log" \
        --memory-csv "${out}/gpu_memory.csv" \
        --warmup "${WARMUP}" \
        --output-json "${out}/bottlenecks.json" \
        --markdown "${out}/bottlenecks.md" || true
    ray stop 2>/dev/null || true
    [[ "${status}" -ne 0 ]] && log "case=${name} FAILED exit=${status}"
    return 0
}

for c in ${CASES}; do
    run_case "${c}"
done

# Roll the per-case JSON into one comparison table.
PROFILE_ROOT="${PROFILE_ROOT}" CASES="${CASES}" python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["PROFILE_ROOT"])
cases = os.environ["CASES"].split()
rows = []
for name in cases:
    path = root / name / "bottlenecks.json"
    if not path.exists():
        rows.append((name, "missing", "", "", "", "")); continue
    data = json.loads(path.read_text())
    avg = data.get("timing", {}).get("avg", {})
    mem = data.get("memory") or {}
    rows.append((
        name,
        f"{avg.get('rollout_time_s', 0):.1f}",
        f"{avg.get('generate_time_s', 0):.1f}",
        f"{avg.get('train_time_s', 0):.1f}",
        f"{avg.get('reward_time_s', 0):.1f}",
        f"{mem.get('peak_gb', 0):.1f}",
    ))

base = None
for r in rows:
    if r[0] == "baseline" and r[1] not in ("missing", "0.0"):
        base = float(r[1]); break

lines = ["# BAGEL-7B Optimization Sweep", "",
         "| case | step_s | generate_s | train_s | reward_s | peak_gb | vs baseline |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    delta = ""
    try:
        if base and r[1] not in ("missing", "0.0"):
            d = (float(r[1]) - base) / base * 100.0
            delta = f"{d:+.1f}%"
    except ValueError:
        pass
    lines.append("| " + " | ".join(r) + f" | {delta} |")
(root / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

log "BAGEL sweep done: ${PROFILE_ROOT}"
