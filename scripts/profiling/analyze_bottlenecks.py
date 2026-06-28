#!/usr/bin/env python
"""Rank UniRL profiling bottlenecks from run logs and recipe constraints.

The analyzer is intentionally log-first: PE/BAGEL profiling runs usually keep
wandb disabled, so the stdout ``run.log`` needs to be enough to recover phase
timings. When phase fields are missing, the analyzer falls back to architecture
signals from the Hydra recipe and the BAGEL profiling report priors.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except Exception:  # pragma: no cover - keeps the tool usable in a bare env.
    DictConfig = ListConfig = None
    OmegaConf = None


_FIELD_ALIASES = {
    "time": "rollout_time_s",
    "step_time": "rollout_time_s",
    "step_time_s": "rollout_time_s",
    "rollout_time_s": "rollout_time_s",
    "generate": "generate_time_s",
    "generate_time_s": "generate_time_s",
    "reward": "reward_time_s",
    "reward_time_s": "reward_time_s",
    "train": "train_time_s",
    "train_time_s": "train_time_s",
    "diff_train": "diffusion_train_time_s",
    "diffusion_train": "diffusion_train_time_s",
    "diffusion_train_time_s": "diffusion_train_time_s",
    "ar_train": "ar_train_time_s",
    "ar_train_time_s": "ar_train_time_s",
    "sync": "weight_sync_time_s",
    "weight_sync": "weight_sync_time_s",
    "weight_sync_time_s": "weight_sync_time_s",
    "diff_sync": "diffusion_weight_sync_time_s",
    "diffusion_weight_sync_time_s": "diffusion_weight_sync_time_s",
    "ar_sync": "ar_weight_sync_time_s",
    "ar_weight_sync_time_s": "ar_weight_sync_time_s",
    "llm_generate": "pe_llm_generate_time_s",
    "pe_llm_generate_time_s": "pe_llm_generate_time_s",
    "diffusion_generate": "pe_diffusion_generate_time_s",
    "pe_diffusion_generate_time_s": "pe_diffusion_generate_time_s",
}

_FIELD_RE = re.compile(r"(?<![\w/.-])([\w/.-]+)[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*s?\b")
_STEP_RE = re.compile(r"(?:step_time|rollout_time_s|time)[=:\s]+([0-9]+(?:\.[0-9]+)?)\s*s?\b")
_PE_TIMING_RE = re.compile(
    r"PEPipeline timing: .*?llm_generate=([0-9.]+)s\s+diffusion_generate=([0-9.]+)s\s+total=([0-9.]+)s"
)


@dataclass
class Bottleneck:
    rank: int
    priority: str
    name: str
    score: float
    impact_pct: float | None
    evidence: str
    recommendation: str


def _canonical_key(raw: str) -> str | None:
    key = raw.strip().strip("\"'")
    if key.startswith("perf/"):
        key = key.removeprefix("perf/")
    return _FIELD_ALIASES.get(key, key if key.endswith("_time_s") else None)


def _coerce_record_from_json(obj: Any) -> dict[str, float]:
    record: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            canonical = _canonical_key(str(key))
            if canonical is not None:
                try:
                    record[canonical] = float(value)
                except (TypeError, ValueError):
                    pass
            if isinstance(value, (dict, list)):
                record.update(_coerce_record_from_json(value))
    elif isinstance(obj, list):
        for value in obj:
            record.update(_coerce_record_from_json(value))
    return record


def parse_timing_records(log_paths: Iterable[Path]) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for path in log_paths:
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            record: dict[str, float] = {}
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    record.update(_coerce_record_from_json(json.loads(stripped)))
                except json.JSONDecodeError:
                    pass

            pe_match = _PE_TIMING_RE.search(line)
            if pe_match:
                record["pe_llm_generate_time_s"] = float(pe_match.group(1))
                record["pe_diffusion_generate_time_s"] = float(pe_match.group(2))
                record["pe_generate_total_time_s"] = float(pe_match.group(3))

            for key, value in _FIELD_RE.findall(line):
                canonical = _canonical_key(key)
                if canonical is not None:
                    record[canonical] = float(value)

            if "rollout_time_s" not in record:
                step_match = _STEP_RE.search(line)
                if step_match:
                    record["rollout_time_s"] = float(step_match.group(1))

            if record:
                records.append(record)
    return records


def parse_log_markers(log_paths: Iterable[Path]) -> dict[str, bool]:
    text = "\n".join(path.read_text(errors="replace") for path in log_paths if path.exists())
    return {
        "saw_rollout_progress": bool(re.search(r"\brollout\s+\d+/\d+\b", text)),
        "saw_sd3_bundle": "role=SD3Bundle" in text,
        "saw_reward_service": "role=RewardService" in text,
        "saw_hf_network_error": "HTTPSConnectionPool(host='huggingface.co'" in text
        or "Network is unreachable" in text,
        "saw_pickscore_processor_error": "Can't load image processor" in text
        and ("CLIP-ViT-H" in text or "laion/" in text),
        "saw_traceback": "Traceback (most recent call last)" in text or "Error executing job" in text,
    }


def summarize_records(records: list[dict[str, float]], warmup: int) -> dict[str, Any]:
    stable = records[warmup:] if len(records) > warmup else records
    keys = sorted({key for rec in stable for key in rec})
    avg: dict[str, float] = {}
    for key in keys:
        values = [rec[key] for rec in stable if key in rec]
        if values:
            avg[key] = statistics.fmean(values)
    return {
        "num_records": len(records),
        "num_stable_records": len(stable),
        "avg": avg,
    }


def _load_cfg(path: Path | None) -> Any:
    if path is None or OmegaConf is None:
        return None
    return OmegaConf.load(path)


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if DictConfig is not None and isinstance(cur, (DictConfig, ListConfig)):
            if part not in cur:
                return default
            cur = cur[part]
        elif isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            return default
    return cur


def _cfg_text(cfg: Any) -> str:
    if cfg is None:
        return ""
    if OmegaConf is not None:
        return OmegaConf.to_yaml(cfg, resolve=False)
    return str(cfg)


def _infer_profile(profile: str, cfg: Any, log_paths: list[Path]) -> str:
    if profile != "auto":
        return profile
    haystack = " ".join([str(path).lower() for path in log_paths]) + " " + _cfg_text(cfg).lower()
    if "pe_trainside" in haystack or "pepipeline" in haystack:
        return "pe"
    if "bagel" in haystack:
        return "bagel"
    if "sd3" in haystack or "stable-diffusion-3.5" in haystack:
        return "sd3"
    return "generic"


def _context_from_cfg(cfg: Any, profile: str) -> dict[str, Any]:
    sampling_prefix = "sampling.diffusion" if _cfg_get(cfg, "sampling.diffusion") is not None else "sampling"
    backend_prefix = "diffusion.backend" if _cfg_get(cfg, "diffusion.backend") is not None else "backend"
    context = {
        "profile": profile,
        "batch_size": _cfg_get(cfg, "batch_size"),
        "num_devices": _cfg_get(cfg, "num_devices"),
        "forward_batch_size": _cfg_get(cfg, "rollout.forward_batch_size"),
        "num_inference_steps": _cfg_get(cfg, f"{sampling_prefix}.num_inference_steps"),
        "samples_per_prompt": _cfg_get(cfg, f"{sampling_prefix}.samples_per_prompt"),
        "root_wrap": _cfg_get(cfg, f"{backend_prefix}.fsdp_cfg.root_wrap", True),
        "forward_prefetch": _cfg_get(cfg, f"{backend_prefix}.fsdp_cfg.forward_prefetch", False),
        "reshard_after_forward": _cfg_get(cfg, f"{backend_prefix}.fsdp_cfg.reshard_after_forward", True),
        "activation_checkpointing": _cfg_get(cfg, f"{backend_prefix}.fsdp_cfg.activation_checkpointing", False),
        "ar_samples_per_prompt": _cfg_get(cfg, "sampling.ar.samples_per_prompt"),
        "diffusion_samples_per_prompt": _cfg_get(cfg, "sampling.diffusion.samples_per_prompt"),
        "ar_max_new_tokens": _cfg_get(cfg, "sampling.ar.max_new_tokens"),
    }
    return {key: value for key, value in context.items() if value is not None}


def _read_memory_csv(path: Path | None, gpu_total_gb: float) -> dict[str, float] | None:
    if path is None:
        return None
    peak_gb = 0.0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(" memory.used [MiB]") or row.get("memory.used [MiB]") or row.get("memory.used")
            if raw is None:
                continue
            match = re.search(r"([0-9.]+)", raw)
            if match:
                peak_gb = max(peak_gb, float(match.group(1)) / 1024.0)
    if peak_gb <= 0:
        return None
    return {
        "peak_gb": peak_gb,
        "total_gb": gpu_total_gb,
        "headroom_pct": max(0.0, (gpu_total_gb - peak_gb) / gpu_total_gb * 100.0),
    }


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator * 100.0


def rank_bottlenecks(
    *,
    profile: str,
    timing: dict[str, Any],
    context: dict[str, Any],
    memory: dict[str, float] | None,
    markers: dict[str, bool],
) -> list[Bottleneck]:
    avg = timing["avg"]
    step = avg.get("rollout_time_s")
    generate = avg.get("generate_time_s") or avg.get("pe_generate_total_time_s")
    reward = avg.get("reward_time_s")
    train = avg.get("train_time_s")
    pe_train = (avg.get("diffusion_train_time_s") or 0.0) + (avg.get("ar_train_time_s") or 0.0)
    train_total = train if train is not None else (pe_train or None)
    generate_pct = _pct(generate, step)
    train_pct = _pct(train_total, step)

    out: list[Bottleneck] = []

    if timing["num_records"] == 0 and not markers.get("saw_rollout_progress"):
        if profile == "pe" and markers.get("saw_pickscore_processor_error"):
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P0",
                    name="PE reward model path is not fully local",
                    score=1000.0,
                    impact_pct=None,
                    evidence="run.log reached RewardService and then tried to fetch PickScore/CLIP files from HuggingFace.",
                    recommendation="Override reward.backend.config.processor_id and model_id to local staged paths.",
                )
            )
        elif profile == "pe" and markers.get("saw_sd3_bundle"):
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P0",
                    name="PE startup blocked in SD3Bundle init",
                    score=1000.0,
                    impact_pct=None,
                    evidence="run.log reached SD3Bundle handle creation but never reached rollout progress/timing.",
                    recommendation="Stage SD3 to local NVMe/cache before profiling, then rerun the PE launcher.",
                )
            )
        elif markers.get("saw_traceback"):
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P0",
                    name="profiling run failed before first rollout",
                    score=1000.0,
                    impact_pct=None,
                    evidence="run.log contains an exception before any timing records.",
                    recommendation="Fix the startup exception before interpreting training bottlenecks.",
                )
            )

    if profile == "bagel":
        fbs = context.get("forward_batch_size")
        steps = context.get("num_inference_steps")
        root_wrap = context.get("root_wrap", True)
        if fbs == 1:
            impact = generate_pct if generate_pct is not None else 70.0
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P0",
                    name="fbs=1 serial rollout",
                    score=0.95,
                    impact_pct=impact,
                    evidence=f"forward_batch_size=1, rollout/generate is {impact:.1f}% of step",
                    recommendation="Fix BAGEL NaViT batch forward so rollout can use fbs>1.",
                )
            )
        if steps:
            impact = generate_pct if generate_pct is not None else 70.0
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P0",
                    name=f"{int(steps)}-step serial denoising",
                    score=0.80,
                    impact_pct=impact,
                    evidence=f"Each sample runs {int(steps)} ordered denoising forwards; this multiplies fbs=1.",
                    recommendation="Validate fewer inference steps, e.g. 14 -> 10, against reward/image quality.",
                )
            )
        if root_wrap is False:
            impact = train_pct if train_pct is not None else 5.0
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P1",
                    name="root_wrap=false blocks forward_prefetch",
                    score=0.35,
                    impact_pct=impact,
                    evidence="BAGEL calls embed/lm_head outside the root forward, so root FSDP wrap is disabled.",
                    recommendation="Refactor BAGEL forward ownership before enabling FSDP forward_prefetch.",
                )
            )
        if memory is not None and memory["headroom_pct"] >= 25.0:
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P0",
                    name="VRAM headroom unused by serial rollout",
                    score=0.30,
                    impact_pct=memory["headroom_pct"],
                    evidence=f"Peak VRAM {memory['peak_gb']:.1f}/{memory['total_gb']:.0f} GB leaves "
                    f"{memory['headroom_pct']:.1f}% headroom.",
                    recommendation="Spend the headroom on batched NaViT forward once fbs>1 is supported.",
                )
            )
        elif memory is None:
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P0",
                    name="VRAM headroom unused by serial rollout",
                    score=0.30,
                    impact_pct=40.0,
                    evidence="No memory CSV supplied; BAGEL report measured roughly 40% H20 VRAM headroom.",
                    recommendation="Re-run with gpu_memory.csv to confirm headroom, then spend it on fbs>1.",
                )
            )

    elif profile == "pe":
        pe_diff = avg.get("pe_diffusion_generate_time_s")
        pe_llm = avg.get("pe_llm_generate_time_s")
        if pe_diff is not None:
            impact = _pct(pe_diff, step) or _pct(pe_diff, generate)
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P0",
                    name="PE SD3 diffusion generate",
                    score=pe_diff,
                    impact_pct=impact,
                    evidence=f"SD3 child generate averages {pe_diff:.2f}s.",
                    recommendation="Tune SD3 inference steps/fbs first; then test compile/prefetch on SD3.",
                )
            )
        if pe_llm is not None:
            impact = _pct(pe_llm, step) or _pct(pe_llm, generate)
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P1",
                    name="PE Qwen rewrite generate",
                    score=pe_llm,
                    impact_pct=impact,
                    evidence=f"LLM rewrite child averages {pe_llm:.2f}s.",
                    recommendation="Reduce max_new_tokens or batch AR generation if rewrite dominates.",
                )
            )
        if train_total is not None:
            out.append(
                Bottleneck(
                    rank=0,
                    priority="P1",
                    name="PE train/backward",
                    score=train_total,
                    impact_pct=train_pct,
                    evidence=f"Train phases average {train_total:.2f}s per step.",
                    recommendation="Check diffusion_train vs ar_train; enable prefetch/compile only after generate is understood.",
                )
            )

    if generate is not None and not any(item.name.endswith("generate") for item in out):
        out.append(
            Bottleneck(
                rank=0,
                priority="P0" if (generate_pct or 0.0) >= 50.0 else "P1",
                name="rollout.generate critical path",
                score=generate,
                impact_pct=generate_pct,
                evidence=f"generate averages {generate:.2f}s per step.",
                recommendation="Break down model-side generate before optimizing train communication.",
            )
        )
    if reward is not None:
        out.append(
            Bottleneck(
                rank=0,
                priority="P2",
                name="reward scoring",
                score=reward,
                impact_pct=_pct(reward, step),
                evidence=f"reward averages {reward:.2f}s per step.",
                recommendation="Only optimize reward if it rises above rollout/train phases.",
            )
        )

    out.sort(key=lambda item: item.score, reverse=True)
    for idx, item in enumerate(out, start=1):
        item.rank = idx
    return out


def write_markdown(path: Path, bottlenecks: list[Bottleneck], summary: dict[str, Any], context: dict[str, Any]) -> None:
    lines = [
        "# Profiling Bottleneck Ranking",
        "",
        "## Timing Summary",
        "",
        f"- records: {summary['num_records']} total, {summary['num_stable_records']} stable",
    ]
    for key, value in sorted(summary["avg"].items()):
        lines.append(f"- {key}: {value:.3f}s")
    if context:
        lines.extend(["", "## Config Context", ""])
        for key, value in sorted(context.items()):
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Bottlenecks",
            "",
            "| rank | priority | bottleneck | impact | evidence | recommendation |",
            "|---:|---|---|---:|---|---|",
        ]
    )
    for item in bottlenecks:
        impact = "" if item.impact_pct is None else f"{item.impact_pct:.1f}%"
        lines.append(
            f"| {item.rank} | {item.priority} | {item.name} | {impact} | "
            f"{item.evidence} | {item.recommendation} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=Path, default=[], help="run.log path; can be repeated")
    parser.add_argument("--config", type=Path, help="Hydra YAML recipe used for the run")
    parser.add_argument("--profile", choices=("auto", "bagel", "pe", "sd3", "generic"), default="auto")
    parser.add_argument("--warmup", type=int, default=3, help="records to skip for stable averages")
    parser.add_argument("--memory-csv", type=Path, help="nvidia-smi CSV sampled during the run")
    parser.add_argument("--gpu-total-gb", type=float, default=97.0, help="per-GPU memory capacity")
    parser.add_argument("--output-json", type=Path, help="write structured JSON report")
    parser.add_argument("--markdown", type=Path, help="write Markdown report")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    profile = _infer_profile(args.profile, cfg, args.log)
    records = parse_timing_records(args.log)
    timing = summarize_records(records, warmup=max(0, args.warmup))
    context = _context_from_cfg(cfg, profile)
    memory = _read_memory_csv(args.memory_csv, args.gpu_total_gb)
    markers = parse_log_markers(args.log)
    bottlenecks = rank_bottlenecks(profile=profile, timing=timing, context=context, memory=memory, markers=markers)
    report = {
        "profile": profile,
        "timing": timing,
        "context": context,
        "memory": memory,
        "markers": markers,
        "bottlenecks": [asdict(item) for item in bottlenecks],
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.markdown:
        write_markdown(args.markdown, bottlenecks, timing, context)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
