"""The backend seam contract — the ``Backend`` protocol + the wire types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

STAGE_KIND_AR = "ar"
STAGE_KIND_DIFFUSION = "diffusion"


@dataclass(frozen=True)
class StageSampling:
    """Sampling-params intent for one stage — kind + plain ctor kwargs."""

    kind: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in (STAGE_KIND_AR, STAGE_KIND_DIFFUSION):
            raise ValueError(
                f"StageSampling.kind must be {STAGE_KIND_AR!r} or {STAGE_KIND_DIFFUSION!r}; got {self.kind!r}"
            )


@dataclass(frozen=True)
class GenerateCall:
    """One ``Omni.generate`` invocation — prompts + per-stage sampling intent."""

    prompts: List[Any]
    sampling: List[StageSampling]
    group_by_request_id: bool = True

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("GenerateCall.prompts must be non-empty")
        if not self.sampling:
            raise ValueError("GenerateCall.sampling must be non-empty")
        if not self.group_by_request_id and len(self.prompts) != 1:
            raise ValueError(
                "GenerateCall.group_by_request_id=False is only valid for "
                f"single-prompt calls; got {len(self.prompts)} prompts"
            )


class OmniRawResult(Protocol):
    """Structural view of vllm-omni's ``OmniRequestOutput`` — the wire fields
    this engine consumes. The native impl passes ``OmniRequestOutput`` through
    (it satisfies this protocol structurally); test fakes (``SimpleNamespace``
    with the fields) stand in. Adapters/utils annotate against it and stay
    vllm-omni-free.

    Population by stage kind:

    - Every output: ``request_id`` (``"{i}_{uuid}"``; the impl consumes it for
      grouping), ``stage_id``, ``final_output_type`` (``"text"`` for the AR
      stage, ``"image"`` / ``"video"`` for the final DiT stage).
    - AR stage (``final_output_type == "text"``): ``request_output`` — the
      nested vLLM ``RequestOutput`` (``.outputs[0].token_ids`` / ``.logprobs``
      / ``.text``) — and ``prompt_token_ids`` (the sample's true, un-padded
      prompt; vLLM runs prompts per-request with no batch padding).
    - DiT stage (``"image"`` / ``"video"``): ``images`` (PIL list; per-prompt
      frame list for video), ``trajectory_latents`` ``[1, T+1, ...]`` (dense —
      every step recorded), ``trajectory_timesteps`` ``[T+1]`` (the field name
      reads "timesteps" but the RL pipeline subclass overwrites its contents
      with the true [0, 1] σ schedule), ``trajectory_log_probs`` ``[1, K]``
      (K = SDE-gated step count; 0 for NFT/forward-process), and
      ``multimodal_output`` — which carries the pipeline's captures under
      ``["metadata"]["unirl"]``, the only capture route that survives the
      worker IPC boundary since vllm-omni 0.26 deleted ``custom_output``
      (read it through ``interception.read_captures``). Documented keys:
      ``"fused_mm_capture"`` (HI3 ``prepare_inputs_for_generation`` capture),
      ``"text_capture"`` (SD3 / HV1.5 ``encode_prompt`` capture),
      ``"sde_step_indices"`` (the SDE-gated step ids echoed by the
      scheduler). Missing capture is a fatal misconfiguration the *adapter*
      raises on — the seam passes the dict through structurally.
    """

    request_id: str
    stage_id: Optional[int]
    final_output_type: Optional[str]
    request_output: Optional[Any]
    prompt_token_ids: Optional[Sequence[int]]
    images: Optional[Sequence[Any]]
    trajectory_latents: Optional["torch.Tensor"]
    trajectory_timesteps: Optional["torch.Tensor"]
    trajectory_log_probs: Optional["torch.Tensor"]
    multimodal_output: Optional[dict]


@runtime_checkable
class Backend(Protocol):
    """The seam every ``vllm_omni`` collaborator reaches the runtime through."""

    def generate(
        self,
        calls: Sequence[GenerateCall],
        *,
        attach_lora: bool = False,
        ar_lora_passthrough: bool = False,
    ) -> List[List[OmniRawResult]]: ...
    def tokenize_prompt(self, text: str, *, task: str, sys_type: str) -> List[int]: ...
    def num_stages(self) -> int: ...
    def tp_per_stage(self) -> Dict[int, int]: ...
    def sleep_task(self) -> None: ...
    def wake_task(self) -> None: ...
    def shutdown(self) -> None: ...
    def ping(self) -> bool: ...
    def update_from_ipc(
        self,
        *,
        peft_config: Optional[dict],
        base_sync_done: bool,
        use_shm: bool,
        replica_rank: Optional[int],
    ) -> None: ...
    def init_weights_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None: ...
    def update_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]],
        flush_cache: bool,
    ) -> None: ...
    def destroy_weights_group(self, *, group_name: str) -> None: ...
    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None: ...
    def set_lora_handle(
        self,
        *,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        peft_config: Optional[dict],
    ) -> None: ...
    def set_lora_copy(
        self,
        *,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        peft_config: Optional[dict],
    ) -> None: ...
    def param_checksums(self, *, names: List[str]) -> dict: ...
    def lora_checksums(self, *, adapter_id: int, names: Optional[List[str]]) -> dict: ...


__all__ = [
    "Backend",
    "GenerateCall",
    "OmniRawResult",
    "StageSampling",
    "STAGE_KIND_AR",
    "STAGE_KIND_DIFFUSION",
]
