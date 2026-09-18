"""Dependency-free contracts spanning sections of a composed Hydra recipe."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from unirl.config.require import require

logger = logging.getLogger(__name__)

ENGINE_PACKAGE = "unirl.rollout.engine."

SYNC_VIA_IPC = "update_weights_from_ipc"
SYNC_VIA_TENSOR = "update_weights_from_tensor"
SYNC_VIA_NCCL = "init_weights_update_group"
SYNC_VIA_LORA = "set_lora_from_tensors"
SYNC_VIA_LORA_COPY = "set_lora_from_tensors_copy"
SYNC_VIA_CHECKPOINT = "update_weights_from_path"

SYNC_RECEIVE_METHODS = (
    SYNC_VIA_IPC,
    SYNC_VIA_TENSOR,
    SYNC_VIA_NCCL,
    SYNC_VIA_LORA,
    SYNC_VIA_LORA_COPY,
    SYNC_VIA_CHECKPOINT,
)

SYNC_LOCAL = "local"
SYNC_REMOTE = "remote"

ENTRYPOINT_AR = "train_ar"
ENTRYPOINT_ASYNC_AR = "train_async_ar"
ENTRYPOINT_DIFFUSION = "train_diffusion"
ENTRYPOINT_ASYNC_DIFFUSION = "train_async_diffusion"
ENTRYPOINT_PE = "train_pe"
ENTRYPOINT_SFT = "train_sft"
ENTRYPOINT_UNIFIED = "train_unified_model"
ENTRYPOINT_AGENTIC = "train_agentic"

KNOWN_ENTRYPOINTS = frozenset(
    {
        ENTRYPOINT_AR,
        ENTRYPOINT_ASYNC_AR,
        ENTRYPOINT_DIFFUSION,
        ENTRYPOINT_ASYNC_DIFFUSION,
        ENTRYPOINT_PE,
        ENTRYPOINT_SFT,
        ENTRYPOINT_UNIFIED,
        ENTRYPOINT_AGENTIC,
    }
)
LAYOUT_ENTRYPOINTS = frozenset({ENTRYPOINT_DIFFUSION, ENTRYPOINT_ASYNC_DIFFUSION})


@dataclass(frozen=True)
class EngineFamily:
    """Recipe-relevant capabilities shared by one rollout-engine package."""

    direct_sampling: bool
    weight_sync: frozenset[str]


_IN_MEMORY_SYNC = frozenset({SYNC_VIA_IPC, SYNC_VIA_TENSOR, SYNC_VIA_NCCL, SYNC_VIA_LORA})

ENGINE_FAMILIES: Mapping[str, EngineFamily] = {
    "trainside": EngineFamily(direct_sampling=True, weight_sync=frozenset()),
    "sglang": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({SYNC_VIA_TENSOR, SYNC_VIA_NCCL, SYNC_VIA_LORA}),
    ),
    "sglang_diffusion": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({SYNC_VIA_TENSOR, SYNC_VIA_NCCL, SYNC_VIA_LORA}),
    ),
    "vllm_omni": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({*_IN_MEMORY_SYNC, SYNC_VIA_LORA_COPY}),
    ),
    "composed": EngineFamily(direct_sampling=False, weight_sync=_IN_MEMORY_SYNC),
    "agentic": EngineFamily(direct_sampling=False, weight_sync=_IN_MEMORY_SYNC),
    "fastvideo": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({SYNC_VIA_CHECKPOINT}),
    ),
}


@dataclass(frozen=True)
class SyncHandler:
    """Receive method and placement boundary owned by a sync-handler class."""

    receive_method: str
    topology: str


SYNC_HANDLERS: Mapping[str, SyncHandler] = {
    "IPCWeightSync": SyncHandler(SYNC_VIA_IPC, SYNC_LOCAL),
    "TensorWeightSync": SyncHandler(SYNC_VIA_TENSOR, SYNC_LOCAL),
    "NCCLWeightSync": SyncHandler(SYNC_VIA_NCCL, SYNC_REMOTE),
    "LocalLoraWeightSync": SyncHandler(SYNC_VIA_LORA, SYNC_LOCAL),
    "RemoteLoraWeightSync": SyncHandler(SYNC_VIA_LORA, SYNC_REMOTE),
    "CheckpointWeightSync": SyncHandler(SYNC_VIA_CHECKPOINT, SYNC_LOCAL),
}

ENGINE_SECTIONS = ("rollout", "ar_rollout", "dit_rollout")
LAYOUTS = ("colocate", "separate")


def engine_family_name(target: str) -> Optional[str]:
    """Return the declared package family for an engine or engine-config target."""
    if not target.startswith(ENGINE_PACKAGE):
        return None
    family = target[len(ENGINE_PACKAGE) :].split(".", 1)[0]
    return family if family in ENGINE_FAMILIES else None


def engine_family(target: str) -> Optional[EngineFamily]:
    """Return the declared capabilities for an engine or engine-config target."""
    family = engine_family_name(target)
    return None if family is None else ENGINE_FAMILIES[family]


@dataclass(frozen=True)
class Block:
    """A targeted recipe block tagged with its source path and sync options."""

    path: str
    target: str
    track: Optional[str] = None
    track_prefix: Optional[str] = None
    copy: bool = False

    @property
    def class_name(self) -> str:
        return self.target.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class RecipeFacts:
    """Normalized facts consumed by every cross-component contract."""

    engines: tuple[Block, ...]
    nested_engines: tuple[Block, ...]
    syncs: tuple[Block, ...]
    has_sync_section: bool
    has_layout: bool
    layout: Optional[str]
    offload: Optional[bool]
    rollout_anchor_device: Any
    freeze_llm: bool

    @classmethod
    def from_cfg(cls, cfg: Any) -> "RecipeFacts":
        """Read normalized facts from a DictConfig or plain mapping."""
        engines = tuple(block for path in ENGINE_SECTIONS if (block := _read_block(cfg, path)) is not None)
        nested_paths = (
            "rollout.config.ar",
            "rollout.config.diffusion",
            "rollout.config.inner",
        )
        nested_engines = tuple(block for path in nested_paths if (block := _read_block_path(cfg, path)) is not None)
        sync_section = _get(cfg, "sync")
        raw_layout = _get(cfg, "layout")
        offload = _get(cfg, "enable_fsdp_offload")
        return cls(
            engines=engines,
            nested_engines=nested_engines,
            syncs=tuple(_read_sync_blocks(sync_section)),
            has_sync_section=sync_section is not None,
            has_layout=_has(cfg, "layout"),
            layout=None if raw_layout is None else str(raw_layout),
            offload=None if offload is None else bool(offload),
            rollout_anchor_device=_get(cfg, "rollout_anchor_device"),
            freeze_llm=bool(_get(cfg, "freeze_llm")),
        )

    def families(self, blocks: Optional[tuple[Block, ...]] = None) -> tuple[tuple[Block, EngineFamily], ...]:
        """Pair known engine blocks with their declared families."""
        known = []
        for block in self.engines if blocks is None else blocks:
            family = engine_family(block.target)
            if family is None:
                logger.info(
                    "cfg.%s._target_=%r has no declared engine family; skipping dependent checks.",
                    block.path,
                    block.target,
                )
                continue
            known.append((block, family))
        return tuple(known)


def _get(cfg: Any, key: str) -> Any:
    """Read one optional mapping key without importing OmegaConf."""
    if cfg is None:
        return None
    getter = getattr(cfg, "get", None)
    return getter(key) if callable(getter) else None


def _has(cfg: Any, key: str) -> bool:
    """Return whether a mapping explicitly contains one key."""
    if cfg is None:
        return False
    try:
        return key in cfg
    except TypeError:
        return False


def _get_path(cfg: Any, path: str) -> Any:
    """Read a dotted mapping path without importing OmegaConf."""
    value = cfg
    for key in path.split("."):
        value = _get(value, key)
    return value


def _read_block(cfg: Any, path: str) -> Optional[Block]:
    target = _get(_get(cfg, path), "_target_")
    return None if target is None else Block(path=path, target=str(target))


def _read_block_path(cfg: Any, path: str) -> Optional[Block]:
    target = _get(_get_path(cfg, path), "_target_")
    return None if target is None else Block(path=path, target=str(target), track=path.rsplit(".", 1)[-1])


def _read_sync_blocks(sync_section: Any) -> list[Block]:
    """Normalize a single handler or per-track handler map without losing track names."""
    if sync_section is None:
        return []
    if (target := _get(sync_section, "_target_")) is not None:
        prefix = _get(sync_section, "track_prefix")
        return [
            Block(
                path="sync",
                target=str(target),
                track_prefix=None if prefix is None else str(prefix),
                copy=bool(_get(sync_section, "copy")),
            )
        ]
    tracks = sync_section.keys() if hasattr(sync_section, "keys") else ()
    blocks = []
    for track in tracks:
        handler = _get(sync_section, track)
        target = _get(handler, "_target_")
        if target is None:
            continue
        prefix = _get(handler, "track_prefix")
        blocks.append(
            Block(
                path=f"sync.{track}",
                target=str(target),
                track=str(track),
                track_prefix=None if prefix is None else str(prefix),
                copy=bool(_get(handler, "copy")),
            )
        )
    return blocks


def _effective_sync_topology(facts: RecipeFacts, entrypoint: Optional[str]) -> str:
    if entrypoint in (ENTRYPOINT_ASYNC_AR, ENTRYPOINT_ASYNC_DIFFUSION):
        return SYNC_REMOTE
    if entrypoint == ENTRYPOINT_AR and facts.rollout_anchor_device is not None:
        return SYNC_REMOTE
    if entrypoint == ENTRYPOINT_UNIFIED and {block.path for block in facts.engines} == {
        "ar_rollout",
        "dit_rollout",
    }:
        return SYNC_REMOTE
    if entrypoint == ENTRYPOINT_DIFFUSION:
        return SYNC_REMOTE if facts.layout == "separate" else SYNC_LOCAL
    if entrypoint is None:
        return SYNC_REMOTE if facts.layout == "separate" else SYNC_LOCAL
    return SYNC_LOCAL


def _nested_engine(facts: RecipeFacts, track: str) -> tuple[Block, ...]:
    suffix = f".{track}"
    return tuple(block for block in facts.nested_engines if block.path.endswith(suffix))


def _sync_engine_blocks(facts: RecipeFacts, sync: Block) -> tuple[Block, ...]:
    rollout = next((block for block in facts.engines if block.path == "rollout"), None)
    family = None if rollout is None else engine_family_name(rollout.target)
    if family == "composed" and sync.track is not None:
        return _nested_engine(facts, sync.track)
    if family == "agentic":
        return _nested_engine(facts, "inner")
    return facts.engines


def _validate_sync_shape(facts: RecipeFacts, *, entrypoint: Optional[str]) -> None:
    """Validate the handler-map shape consumed by each trainer."""
    if entrypoint == ENTRYPOINT_PE:
        required = {"diffusion"} if facts.freeze_llm else {"ar", "diffusion"}
        actual = {sync.track for sync in facts.syncs}
        require(
            actual == required, f"train_pe requires sync tracks {sorted(required)}; found {sorted(actual, key=str)}."
        )
        for sync in facts.syncs:
            require(
                sync.track_prefix == sync.track,
                f"cfg.{sync.path}.track_prefix must be {sync.track!r}; got {sync.track_prefix!r}.",
            )
            require(
                bool(_nested_engine(facts, str(sync.track))),
                f"cfg.rollout.config.{sync.track} must declare an engine config for cfg.{sync.path}.",
            )
        return
    require(
        all(sync.track is None for sync in facts.syncs),
        f"{entrypoint or 'this recipe'} expects one cfg.sync handler, not a per-track map.",
    )
    if entrypoint == ENTRYPOINT_AGENTIC:
        require(bool(_nested_engine(facts, "inner")), "cfg.rollout.config.inner must declare an engine config.")


def _validate_engine_shape(facts: RecipeFacts, *, entrypoint: Optional[str]) -> None:
    """Require the exact engine sections consumed by each entrypoint."""
    paths = {block.path for block in facts.engines}
    if entrypoint == ENTRYPOINT_UNIFIED:
        require(
            paths in ({"rollout"}, {"ar_rollout", "dit_rollout"}),
            "train_unified_model requires either cfg.rollout or both cfg.ar_rollout and cfg.dit_rollout.",
        )
    elif entrypoint == ENTRYPOINT_SFT:
        require(not paths, "train_sft does not consume rollout engine sections.")
    elif entrypoint in KNOWN_ENTRYPOINTS:
        require(paths == {"rollout"}, f"{entrypoint} requires exactly one cfg.rollout engine section.")


def _validate_handler_topology(facts: RecipeFacts, *, entrypoint: Optional[str]) -> None:
    topology = _effective_sync_topology(facts, entrypoint)
    for sync in facts.syncs:
        handler = SYNC_HANDLERS.get(sync.class_name)
        if handler is None:
            continue
        require(
            handler.topology == topology,
            f"cfg.{sync.path}={sync.class_name} is a {handler.topology}-engine handler, but "
            f"{entrypoint or 'this recipe'} wires rollout through the {topology} boundary.",
        )
        if entrypoint == ENTRYPOINT_ASYNC_AR:
            require(sync.class_name == "NCCLWeightSync", "train_async_ar supports only NCCLWeightSync.")
        if entrypoint == ENTRYPOINT_AGENTIC:
            require(sync.class_name == "TensorWeightSync", "train_agentic supports only TensorWeightSync.")
        if entrypoint == ENTRYPOINT_AR and facts.rollout_anchor_device is not None:
            require(
                sync.class_name == "RemoteLoraWeightSync",
                "anchored train_ar rollout supports only RemoteLoraWeightSync.",
            )
        if entrypoint == ENTRYPOINT_UNIFIED and len(facts.engines) == 2:
            require(
                sync.class_name == "RemoteLoraWeightSync",
                "two-engine train_unified_model supports only RemoteLoraWeightSync.",
            )


def is_direct_sampling(cfg: Any) -> bool:
    """Return whether every recognized top-level engine samples in train actors."""
    families = RecipeFacts.from_cfg(cfg).families()
    return bool(families) and all(family.direct_sampling for _, family in families)


def validate_weight_sync_contract(cfg: Any, *, entrypoint: Optional[str] = None) -> None:
    """Validate sampling mode, handler shape, topology, and receiver capabilities."""
    facts = RecipeFacts.from_cfg(cfg)
    _validate_engine_shape(facts, entrypoint=entrypoint)
    families = facts.families()
    if not families:
        if entrypoint in KNOWN_ENTRYPOINTS:
            require(not facts.has_sync_section, f"{entrypoint} has no recognized rollout engine but declares cfg.sync.")
        return

    direct = [block.path for block, family in families if family.direct_sampling]
    dedicated = [block.path for block, family in families if not family.direct_sampling]
    require(
        not (direct and dedicated),
        f"recipe mixes direct engines at cfg.{', cfg.'.join(direct)} with dedicated engines at "
        f"cfg.{', cfg.'.join(dedicated)}.",
    )

    if direct:
        found = _describe(facts.syncs) or "a handler-less sync section"
        require(
            not facts.has_sync_section,
            f"cfg.{direct[0]} samples on the train actors and cannot use cfg.sync; found {found}.",
        )
        return

    if entrypoint == ENTRYPOINT_UNIFIED and [block.path for block in facts.engines] == ["rollout"]:
        raise ValueError(
            "train_unified_model single-engine mode does not wire weight sync for a dedicated rollout engine; "
            "use the trainside engine or the ar_rollout + dit_rollout mode."
        )

    require(
        bool(facts.syncs),
        f"cfg.{dedicated[0]} runs a dedicated rollout copy and requires a cfg.sync handler.",
    )
    _validate_sync_shape(facts, entrypoint=entrypoint)
    _validate_handler_topology(facts, entrypoint=entrypoint)

    for sync in facts.syncs:
        handler = SYNC_HANDLERS.get(sync.class_name)
        if handler is None:
            continue
        needed = (
            SYNC_VIA_LORA_COPY if sync.class_name == "RemoteLoraWeightSync" and sync.copy else handler.receive_method
        )
        for block, family in facts.families(_sync_engine_blocks(facts, sync)):
            require(
                needed in family.weight_sync,
                f"cfg.{sync.path}={sync.class_name} needs {needed}(), but "
                f"cfg.{block.path}._target_={block.target!r} supports {sorted(family.weight_sync)}.",
            )


def validate_rollout_layout(cfg: Any, *, entrypoint: Optional[str] = None) -> None:
    """Validate rollout layout values only where the selected entrypoint consumes them."""
    facts = RecipeFacts.from_cfg(cfg)
    if facts.has_layout:
        require(
            facts.layout is not None and facts.layout in LAYOUTS,
            f"cfg.layout={facts.layout!r} is invalid; expected one of {list(LAYOUTS)}.",
        )
    if entrypoint is not None and entrypoint not in LAYOUT_ENTRYPOINTS:
        require(facts.layout is None, f"{entrypoint} does not consume cfg.layout; remove it.")

    if entrypoint == ENTRYPOINT_ASYNC_DIFFUSION:
        require(
            facts.layout in (None, "separate"),
            f"train_async_diffusion requires cfg.layout='separate'; got {facts.layout!r}.",
        )
        effective = "separate"
    else:
        effective = facts.layout or "colocate"
    if effective != "separate":
        return
    for block, family in facts.families():
        require(
            not family.direct_sampling,
            f"cfg.layout='separate' cannot place direct-sampling cfg.{block.path}._target_={block.target!r}.",
        )


def validate_offload_contract(cfg: Any, *, entrypoint: Optional[str] = None) -> None:
    """Reject an explicit FSDP-offload request with direct sampling."""
    del entrypoint
    facts = RecipeFacts.from_cfg(cfg)
    if facts.offload is not True:
        return
    for block, family in facts.families():
        require(
            not family.direct_sampling,
            f"cfg.enable_fsdp_offload=true is incompatible with direct-sampling "
            f"cfg.{block.path}._target_={block.target!r}.",
        )


CONTRACTS = (
    validate_weight_sync_contract,
    validate_rollout_layout,
    validate_offload_contract,
)


def validate_recipe(cfg: Any, *, entrypoint: str) -> None:
    """Run every cross-component contract before trainer construction."""
    require(entrypoint in KNOWN_ENTRYPOINTS, f"unknown training entrypoint {entrypoint!r}")
    for contract in CONTRACTS:
        try:
            contract(cfg, entrypoint=entrypoint)
        except ValueError as exc:
            raise ValueError(f"{entrypoint}: invalid recipe. {exc}") from exc


def _describe(blocks: tuple[Block, ...]) -> str:
    return ", ".join(f"cfg.{block.path}={block.class_name}" for block in blocks)


__all__ = [
    "Block",
    "CONTRACTS",
    "ENGINE_FAMILIES",
    "ENGINE_PACKAGE",
    "ENGINE_SECTIONS",
    "EngineFamily",
    "KNOWN_ENTRYPOINTS",
    "LAYOUTS",
    "RecipeFacts",
    "SYNC_HANDLERS",
    "SYNC_RECEIVE_METHODS",
    "SYNC_VIA_CHECKPOINT",
    "SYNC_VIA_IPC",
    "SYNC_VIA_LORA",
    "SYNC_VIA_LORA_COPY",
    "SYNC_VIA_NCCL",
    "SYNC_VIA_TENSOR",
    "SyncHandler",
    "engine_family",
    "engine_family_name",
    "is_direct_sampling",
    "validate_offload_contract",
    "validate_recipe",
    "validate_rollout_layout",
    "validate_weight_sync_contract",
]
