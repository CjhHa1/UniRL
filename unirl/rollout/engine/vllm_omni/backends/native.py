"""The native ``Backend`` impl — the in-process ``Omni`` orchestrator."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pprint import pformat
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    import torch

from unirl.rollout.engine.vllm_omni.backends.base import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.utils.graceful_shutdown import terminate_descendants

logger = logging.getLogger(__name__)

_ENGINE_PROC_PREFIX = "VLLM::"


@contextmanager
def _preserve_cudnn_sdp_state():
    """Restore the process-wide cuDNN SDPA setting after vLLM imports."""
    import torch

    enabled = torch.backends.cuda.cudnn_sdp_enabled() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        if enabled is not None:
            torch.backends.cuda.enable_cudnn_sdp(enabled)


def _import_omni_runtime() -> Dict[str, Any]:
    """Lazy import of the vllm-omni runtime types. Imported once per process."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams as VLLMSamplingParams
    from vllm_omni.diffusion.data import OmniSleepTask, OmniWakeTask
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

    return {
        "AutoTokenizer": AutoTokenizer,
        "Omni": Omni,
        "OmniDiffusionSamplingParams": OmniDiffusionSamplingParams,
        "OmniLoRARequest": OmniLoRARequest,
        "OmniSleepTask": OmniSleepTask,
        "OmniWakeTask": OmniWakeTask,
        "VLLMSamplingParams": VLLMSamplingParams,
    }


def _resolve_deploy_config(name: str) -> str:
    """Return the absolute path of the local deploy-config YAML asset."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "deploy_configs", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"_resolve_deploy_config: YAML not found at {path}")
    return path


@contextmanager
def _master_port_env(port: Optional[int]):
    """Pin ``MASTER_PORT`` to the reserved engine port for the duration of ``Omni()``."""
    if port is None:
        yield
        return
    saved = os.environ.get("MASTER_PORT")
    os.environ["MASTER_PORT"] = str(port)
    logger.info("VLLM-Omni boot: MASTER_PORT=%s (was %s) for engine construction", port, saved)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = saved


class VLLMOmniBackend:
    """The native ``Backend`` impl over the ``Omni`` orchestrator."""

    def __init__(
        self,
        omni: Any,
        runtime: Dict[str, Any],
        *,
        tokenizer: Optional[Any],
    ) -> None:
        self._omni: Optional[Any] = omni
        self._rt = runtime
        self._tokenizer = tokenizer

    @classmethod
    def boot(cls, intent: Dict[str, Any]) -> "VLLMOmniBackend":
        """Spell the intent into ``Omni`` ctor kwargs and spawn."""
        import multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # vLLM disables cuDNN SDPA process-wide at import. Preserve the
        # colocated train actor's state; spawned workers initialize their own.
        with _preserve_cudnn_sdp_state():
            from unirl.rollout.engine.vllm_omni.patches import install as install_patches
            from unirl.rollout.engine.vllm_omni.plugin import register_unirl_runtime

            install_patches()
            rt = _import_omni_runtime()
            register_unirl_runtime()

        if intent.get("clear_cuda_visible"):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        import fcntl

        # Release the trainer CUDA cache before spawning the engine process.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - belt and braces; never block a boot
            pass

        deploy_config_path = _resolve_deploy_config(intent["deploy_config"])
        omni_kwargs = dict(intent["omni_kwargs"])
        if intent.get("enable_sleep_mode"):
            omni_kwargs["enable_sleep_mode"] = True
        ports = intent.get("ports")
        boot_master_port = ports.master_port if ports is not None else None
        logger.info(
            "VLLM-Omni boot intent (before engine startup):\n%s",
            pformat(
                {
                    **intent,
                    "deploy_config_path": deploy_config_path,
                    "assembled_omni_kwargs": omni_kwargs,
                    "boot_master_port": boot_master_port,
                },
                sort_dicts=True,
            ),
        )
        serialize = os.environ.get("UNIRL_OMNI_BOOT_SERIALIZE", "1") != "0"
        lock_file = open("/tmp/unirl_omni_boot.lock", "a+") if serialize else None
        omni = None
        try:
            try:
                if lock_file is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                with _master_port_env(boot_master_port):
                    omni = rt["Omni"](
                        model=intent["model_path"],
                        deploy_config=deploy_config_path,
                        **omni_kwargs,
                    )
            finally:
                if lock_file is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    lock_file.close()

            stage_configs = omni.engine.stage_configs
            try:
                from omegaconf import OmegaConf

                resolved_stage_configs = OmegaConf.to_container(
                    OmegaConf.create(stage_configs),
                    resolve=True,
                )
            except Exception:  # noqa: BLE001 - config logging must never block boot
                resolved_stage_configs = stage_configs
            logger.info(
                "VLLM-Omni resolved runtime stage configs (after all overrides):\n%s",
                pformat(resolved_stage_configs, sort_dicts=True),
            )

            tokenizer = None
            if intent.get("needs_driver_tokenizer"):
                tokenizer = rt["AutoTokenizer"].from_pretrained(intent["model_path"], trust_remote_code=True)

            return cls(
                omni,
                rt,
                tokenizer=tokenizer,
            )
        except BaseException:
            logger.exception("VLLM-Omni boot failed; tearing down any engine processes")
            if omni is not None:
                try:
                    omni.close()
                except Exception:
                    logger.exception("Failed to close the half-booted vLLM-Omni engine")
            terminate_descendants(os.getpid(), name_prefix=_ENGINE_PROC_PREFIX)
            raise

    def _require_omni(self) -> Any:
        if self._omni is None:
            raise RuntimeError("VLLMOmniBackend: engine not initialized (shut down?)")
        return self._omni

    def generate(
        self,
        calls: Sequence[GenerateCall],
        *,
        attach_lora: bool = False,
        ar_lora_passthrough: bool = False,
    ) -> List[List[OmniRawResult]]:
        """Run each call through ``Omni.generate`` and group per request."""
        omni = self._require_omni()
        groups: List[List[OmniRawResult]] = []
        for call in calls:
            sp_list = [self._build_sampling_params(s, attach_lora=attach_lora) for s in call.sampling]
            generate_kwargs: Dict[str, Any] = {"use_tqdm": False}
            if attach_lora and ar_lora_passthrough:
                generate_kwargs["lora_request"] = self._lora_request()
            flat = list(omni.generate(call.prompts, sp_list, **generate_kwargs))
            if call.group_by_request_id:
                groups.extend(_group_by_request(flat, len(call.prompts)))
            else:
                groups.append(flat)
        return groups

    def _build_sampling_params(self, sampling: StageSampling, *, attach_lora: bool) -> Any:
        if sampling.kind == STAGE_KIND_AR:
            return self._rt["VLLMSamplingParams"](**sampling.kwargs)
        sp = self._rt["OmniDiffusionSamplingParams"](**sampling.kwargs)
        if attach_lora:
            sp.lora_request = self._lora_request()
            sp.lora_scale = 1.0
        return sp

    def _lora_request(self) -> Any:
        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            UNIRL_LORA_INT_ID,
            UNIRL_LORA_NAME,
            UNIRL_LORA_PATH,
        )

        return self._rt["OmniLoRARequest"](
            lora_name=UNIRL_LORA_NAME,
            lora_int_id=UNIRL_LORA_INT_ID,
            lora_path=UNIRL_LORA_PATH,
        )

    def tokenize_prompt(self, text: str, *, task: str, sys_type: str) -> List[int]:
        """HI3 prompt tokens via vllm-omni's ``build_prompt_tokens``."""
        if self._tokenizer is None:
            raise RuntimeError(
                "VLLMOmniBackend.tokenize_prompt: no driver tokenizer loaded "
                "(boot intent did not set needs_driver_tokenizer)."
            )
        from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
            build_prompt_tokens,
        )

        # vLLM-Omni returns a PromptTokensResult here, not a bare id list.
        return list(build_prompt_tokens(text, self._tokenizer, task=task, sys_type=sys_type).token_ids)

    def num_stages(self) -> int:
        return int(self._require_omni().engine.num_stages)

    def tp_per_stage(self) -> Dict[int, int]:
        tp_map: Dict[int, int] = {}
        for entry in self._require_omni().engine.stage_configs:
            engine_args = entry.engine_args
            tp = engine_args.get("tensor_parallel_size")
            if tp is None:
                parallel_config = engine_args.get("parallel_config")
                if parallel_config is not None:
                    tp = parallel_config.get("tensor_parallel_size")
            tp_map[int(entry.stage_id)] = int(tp) if tp is not None else 1
        return tp_map

    @staticmethod
    def _require_rpc_success(
        action: str,
        stage_id: int,
        results: object,
        *,
        allow_false: bool = False,
    ) -> None:
        """Fail closed when a stage or replica reports an unsupported/failed RPC."""

        def validate(result: object) -> None:
            if result is None:
                return
            if isinstance(result, (list, tuple)):
                for item in result:
                    validate(item)
                return
            if isinstance(result, bool):
                if not result and not allow_false:
                    raise RuntimeError(f"vllm-omni {action} returned False on stage {stage_id}")
                return
            if not isinstance(result, Mapping):
                raise RuntimeError(f"vllm-omni {action} returned an unexpected result on stage {stage_id}: {result!r}")

            status = result.get("status")
            error = result.get("error")
            if result.get("supported") is False or error or (status is not None and status != "SUCCESS"):
                reason = result.get("reason") or result.get("error_msg") or error or status
                raise RuntimeError(f"vllm-omni {action} failed on stage {stage_id}: {reason!r}")

        validate(results)

    def _collective_rpc(
        self,
        stage_id: int,
        method: str,
        *,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        allow_false: bool = False,
    ) -> List[Any]:
        results = self._require_omni().engine.collective_rpc(
            method=method,
            args=args,
            kwargs=kwargs,
            stage_ids=[stage_id],
        )
        self._require_rpc_success(method, stage_id, results, allow_false=allow_false)
        return results

    @staticmethod
    def _require_ack_success(action: str, stage_id: int, task_id: str, acks: object) -> None:
        # Worker handlers catch their own exceptions and answer
        # ``OmniACK(status="ERROR")`` instead of raising, so a discarded return
        # value turns a failed sleep/wake into a silent partial transition — the
        # exact state the engine's transition latch exists to refuse.
        success_count = 0

        def validate_result(result: Any) -> None:
            nonlocal success_count
            if result is None:
                # Non-reporting worker ranks legitimately return None; at least
                # one explicit rank-0 SUCCESS ACK is still required below.
                return
            if isinstance(result, (list, tuple)):
                for item in result:
                    validate_result(item)
                return

            status = result.status
            if status is None:
                raise RuntimeError(f"vllm-omni {action} returned no ACK status for stage {stage_id}: result={result!r}")
            if status != "SUCCESS":
                raise RuntimeError(
                    f"vllm-omni {action} failed on stage {stage_id}: worker rank "
                    f"{result.rank} answered status={status!r} "
                    f"error={result.error_msg!r}"
                )
            if result.stage_id != stage_id:
                raise RuntimeError(
                    f"vllm-omni {action} ACK stage mismatch: expected {stage_id}, got {result.stage_id!r}"
                )
            if result.task_id != task_id:
                raise RuntimeError(
                    f"vllm-omni {action} ACK task mismatch on stage {stage_id}: "
                    f"expected {task_id!r}, got {result.task_id!r}"
                )
            success_count += 1

        validate_result(acks)
        if success_count == 0:
            raise RuntimeError(f"vllm-omni {action} returned no successful ACK for stage {stage_id}")

    def sleep_task(self) -> None:
        """Sleep AR stages through EngineCore and diffusion stages through worker RPC."""
        import uuid

        omni = self._require_omni()
        for sid in range(self.num_stages()):
            if omni.engine.get_stage_metadata(sid).stage_type != "diffusion":
                self._collective_rpc(sid, "sleep", args=(1, "abort"))
                continue
            task_id = str(uuid.uuid4())
            acks = omni.engine.collective_rpc(
                method="handle_sleep_task",
                args=(self._rt["OmniSleepTask"](level=1, task_id=task_id),),
                stage_ids=[sid],
            )
            self._require_ack_success("sleep", sid, task_id, acks)

    def wake_task(self) -> None:
        """Wake AR stages through EngineCore and diffusion stages through worker RPC."""
        import uuid

        import torch

        omni = self._require_omni()
        for sid in range(self.num_stages()):
            if omni.engine.get_stage_metadata(sid).stage_type != "diffusion":
                # ``None`` wakes every tag, including UniRL's MoE workspace tag.
                self._collective_rpc(sid, "wake_up", kwargs={"tags": None})
                continue
            task_id = str(uuid.uuid4())
            acks = omni.engine.collective_rpc(
                method="handle_wake_task",
                args=(self._rt["OmniWakeTask"](tags=None, task_id=task_id),),
                stage_ids=[sid],
            )
            self._require_ack_success("wake", sid, task_id, acks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def ping(self) -> bool:
        omni = self._omni
        if omni is None:
            return False
        try:
            return omni.is_running
        except Exception:
            logger.exception("vLLM-Omni health check failed")
            return False

    def shutdown(self) -> None:
        if self._omni is not None:
            try:
                self._omni.close()
            finally:
                self._omni = None

        reaped = terminate_descendants(os.getpid(), name_prefix=_ENGINE_PROC_PREFIX)
        if reaped:
            logger.warning("Reaped %d engine process(es) that outlived the vLLM-Omni shutdown", reaped)

    def update_from_ipc(
        self,
        *,
        peft_config: Optional[dict],
        base_sync_done: bool,
        use_shm: bool,
        replica_rank: Optional[int],
    ) -> None:
        """Fan a bucketed CUDA-IPC state-dict update out to per-stage workers."""
        kwargs = {
            "peft_config": peft_config,
            "base_sync_done": base_sync_done,
            "use_shm": use_shm,
            "replica_rank": replica_rank,
        }
        for sid in range(self.num_stages()):
            self._collective_rpc(
                sid,
                "update_weights_from_ipc",
                kwargs={**kwargs, "stage_id": sid},
            )

    def init_weights_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        kwargs = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": rank_offset,
            "world_size": world_size,
            "group_name": group_name,
            "backend": backend,
        }
        for sid in range(self.num_stages()):
            self._collective_rpc(
                sid,
                "init_weights_update_group",
                kwargs=kwargs,
            )

    def update_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]],
        flush_cache: bool,
    ) -> None:
        kwargs = {
            "names": list(names),
            "dtypes": list(dtypes),
            "shapes": [list(s) for s in shapes],
            "group_name": group_name,
            "target_modules": list(target_modules) if target_modules else None,
            "flush_cache": flush_cache,
        }
        for sid in range(self.num_stages()):
            self._collective_rpc(
                sid,
                "update_weights_from_distributed",
                kwargs=kwargs,
            )

    def destroy_weights_group(self, *, group_name: str) -> None:
        if self._omni is None:
            return
        for sid in range(self.num_stages()):
            self._collective_rpc(
                sid,
                "destroy_weights_update_group",
                kwargs={"group_name": group_name},
            )

    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None:
        """Fan a SGLang-shape tensor payload to per-stage workers."""
        kwargs = {
            "serialized_named_tensors": list(serialized_named_tensors),
            "target_modules": list(target_modules) if target_modules else None,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        for sid in range(self.num_stages()):
            self._collective_rpc(
                sid,
                "update_weights_from_tensor",
                kwargs=kwargs,
            )

    def set_lora_handle(
        self,
        *,
        lora_tensors: Dict[str, torch.Tensor],
        peft_config: Optional[dict],
    ) -> None:
        """Zero-copy LoRA push via ``MultiprocessingSerializer`` shm handles."""
        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            UNIRL_LORA_INT_ID,
            UNIRL_LORA_NAME,
            UNIRL_LORA_PATH,
        )
        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            MultiprocessingSerializer,
        )
        from unirl.utils.peft_merge import adapt_lora_for_vllm

        lora_tensors = adapt_lora_for_vllm(lora_tensors)
        self._remove_existing_lora(UNIRL_LORA_INT_ID)
        peft_dict = {} if peft_config is None else dict(peft_config)

        for sid in range(self.num_stages()):
            cloned = {name: t.detach().clone() for name, t in lora_tensors.items()}
            serialized = MultiprocessingSerializer.serialize(cloned, output_str=True)
            self._collective_rpc(
                sid,
                "set_lora_from_tensor_dict",
                args=(
                    UNIRL_LORA_NAME,
                    UNIRL_LORA_INT_ID,
                    UNIRL_LORA_PATH,
                    peft_dict,
                    serialized,
                ),
            )

    def set_lora_copy(
        self,
        *,
        lora_tensors: Dict[str, torch.Tensor],
        peft_config: Optional[dict],
    ) -> None:
        """Byte-copy LoRA push (``torch.save`` + base64) — TP>1-broadcast-safe."""
        import base64
        import io

        import torch

        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            UNIRL_LORA_INT_ID,
            UNIRL_LORA_NAME,
            UNIRL_LORA_PATH,
        )
        from unirl.utils.peft_merge import adapt_lora_for_vllm

        lora_tensors = adapt_lora_for_vllm(lora_tensors)
        self._remove_existing_lora(UNIRL_LORA_INT_ID)
        peft_dict = {} if peft_config is None else dict(peft_config)

        cpu_tensors = {name: t.detach().to("cpu") for name, t in lora_tensors.items()}
        buf = io.BytesIO()
        torch.save(cpu_tensors, buf)
        serialized = base64.b64encode(buf.getvalue()).decode("ascii")

        for sid in range(self.num_stages()):
            self._collective_rpc(
                sid,
                "set_lora_from_tensor_dict_copy",
                args=(
                    UNIRL_LORA_NAME,
                    UNIRL_LORA_INT_ID,
                    UNIRL_LORA_PATH,
                    peft_dict,
                    serialized,
                ),
            )

    def _remove_existing_lora(self, adapter_id: int) -> None:
        """Drop the existing adapter on every stage before re-adding."""
        for sid in range(self.num_stages()):
            self._collective_rpc(
                sid,
                "remove_lora",
                args=(adapter_id,),
                allow_false=True,
            )

    def param_checksums(self, *, names: List[str]) -> dict:
        """Fan ``_unirl_loaded_param_checksums`` across stages and ranks."""
        out: dict = {}
        for sid in range(self.num_stages()):
            results = self._collective_rpc(
                sid,
                "_unirl_loaded_param_checksums",
                args=(list(names),),
            )
            out[sid] = results[0] if isinstance(results, list) and results else results
        return out

    def lora_checksums(self, *, adapter_id: int, names: Optional[List[str]]) -> dict:
        """Fan ``_unirl_loaded_lora_checksums`` across stages and ranks."""
        out: dict = {}
        for sid in range(self.num_stages()):
            results = self._collective_rpc(
                sid,
                "_unirl_loaded_lora_checksums",
                args=(adapter_id, list(names) if names else None),
            )
            out[sid] = results[0] if isinstance(results, list) and results else results
        return out


def _group_by_request(flat_outputs: Sequence[Any], n: int) -> List[List[Any]]:
    """Group ``Omni.generate``'s flat output list into per-request lists."""
    grouped: List[List[Any]] = [[] for _ in range(n)]
    for out in flat_outputs:
        rid = out.request_id
        try:
            idx_part, suffix = rid.split("_", 1)
            if not suffix:
                raise ValueError("empty request-id suffix")
            idx = int(idx_part)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"vllm-omni returned an invalid request_id: {rid!r}") from exc
        if not 0 <= idx < n:
            raise RuntimeError(f"vllm-omni request_id index {idx} is outside the prompt batch of size {n}")
        grouped[idx].append(out)
    return grouped


__all__ = ["VLLMOmniBackend"]
