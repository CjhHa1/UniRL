"""Weight sync — the canonical sync ops + LoRA lifecycle, owned by one component."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from unirl.rollout.engine.vllm_omni.backends import Backend

logger = logging.getLogger(__name__)


class WeightSync:
    """Sync ops + LoRA lifecycle over the seam (one instance per engine)."""

    def __init__(
        self,
        backend: Backend,
        *,
        lora_copy_transport: bool,
    ) -> None:
        self._backend = backend
        self._lora_copy_transport = lora_copy_transport
        self._lora_loaded = False
        self._last_lora_name: Optional[str] = None
        self._last_lora_tensors: Optional[Dict[str, torch.Tensor]] = None
        self._last_peft_config: Optional[dict] = None

    @property
    def lora_loaded(self) -> bool:
        """True when a pushed adapter should be activated on generate."""
        return self._lora_loaded

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        replica_rank: Optional[int] = None,
    ) -> None:
        self._backend.update_from_ipc(
            peft_config=peft_config,
            base_sync_done=base_sync_done,
            use_shm=use_shm,
            replica_rank=replica_rank,
        )
        if peft_config and base_sync_done:
            self._lora_loaded = True

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        self._backend.init_weights_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
    ) -> None:
        self._backend.update_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            target_modules=target_modules,
            flush_cache=flush_cache,
        )

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        self._backend.destroy_weights_group(group_name=group_name)

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        self._backend.update_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
        )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Hot-swap the adapter via the zero-copy shm-handle transport."""
        self._cache_lora(adapter_name, lora_tensors, peft_config)
        self._backend.set_lora_handle(lora_tensors=lora_tensors, peft_config=peft_config)
        self._lora_loaded = True

    def set_lora_from_tensors_copy(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Hot-swap the adapter via the TP>1-safe byte-copy transport."""
        self._cache_lora(adapter_name, lora_tensors, peft_config)
        self._backend.set_lora_copy(lora_tensors=lora_tensors, peft_config=peft_config)
        self._lora_loaded = True

    def _cache_lora(
        self, adapter_name: str, lora_tensors: Dict[str, torch.Tensor], peft_config: Optional[dict]
    ) -> None:
        """Clone the adapter state so a sleep/wake cycle can re-push it."""
        self._last_lora_name = adapter_name
        self._last_lora_tensors = {name: t.detach().clone() for name, t in lora_tensors.items()}
        self._last_peft_config = {} if peft_config is None else dict(peft_config)

    def loaded_param_checksums(self, *, names: List[str]) -> dict:
        return self._backend.param_checksums(names=names)

    def loaded_lora_checksums(self, *, adapter_id: int, names: Optional[List[str]] = None) -> dict:
        return self._backend.lora_checksums(adapter_id=adapter_id, names=names)

    def restore_lora_after_wake(self) -> None:
        """Re-push the cached adapter after a wake (v1 parity)."""
        if self._last_lora_tensors is None:
            return
        logger.info(
            "[LoRA-WAKE] Re-loading LoRA after sleep/wake. adapter_name=%s",
            self._last_lora_name,
        )
        try:
            if self._lora_copy_transport:
                self.set_lora_from_tensors_copy(
                    self._last_lora_name,
                    self._last_lora_tensors,
                    peft_config=self._last_peft_config,
                )
            else:
                self.set_lora_from_tensors(
                    self._last_lora_name,
                    self._last_lora_tensors,
                    peft_config=self._last_peft_config,
                )
        except Exception as exc:
            self._lora_loaded = False
            raise RuntimeError(
                f"[LoRA-WAKE] Failed to re-load LoRA adapter "
                f"{self._last_lora_name!r} after sleep/wake; refusing "
                f"to continue serving because rollout would silently "
                f"run the base model, drifting old/new log-probs and "
                f"the GRPO ratio. Original error: {exc!r}"
            ) from exc
        logger.info("[LoRA-WAKE] LoRA re-loaded successfully.")


__all__ = ["WeightSync"]
