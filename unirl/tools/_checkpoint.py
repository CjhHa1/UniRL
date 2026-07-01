"""Offline reader for the two checkpoint flavors ``BaseFSDP2Backend.save`` writes.

Both export tools (:mod:`unirl.tools.export_full`, :mod:`unirl.tools.export_adapter`)
consume a checkpoint as a flat dict with ``policy_state_dict`` (plus
``lora_config`` / ``step`` / ``save_mode`` when present). The legacy ``torch``
format already is that dict; the sharded ``dcp`` format is reassembled here in a
single process — no distributed group, since the export tools run on one host.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict

import torch


def load_training_checkpoint(path: str) -> Dict[str, object]:
    """Load a UniRL checkpoint (``torch`` or ``dcp``) into the legacy dict shape.

    ``path`` is the ``checkpoint-<step>`` directory (either format) or, for the
    ``torch`` format, the ``checkpoint.pt`` file itself. A directory holding a
    DCP ``.metadata`` is read as sharded; anything else falls back to the legacy
    single-file pickle.
    """
    if os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata")):
        return _load_dcp(path)
    file_path = os.path.join(path, "checkpoint.pt") if os.path.isdir(path) else path
    return _torch_load(file_path)


def _torch_load(file_path: str) -> Dict[str, object]:
    # Prefer the safe unpickler; fall back for older checkpoints that carry
    # pickled (non-tensor) objects it rejects.
    try:
        return torch.load(file_path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        return torch.load(file_path, map_location="cpu")
    except RuntimeError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(file_path, map_location="cpu")


def _load_dcp(path: str) -> Dict[str, object]:
    # Single-process reassembly of the sharded save: the empty-state-dict planner
    # reads the global tensors from every shard (no model and no process group
    # needed). The app-level metadata.pt carries lora_config / step / save_mode
    # beside DCP's own ``.metadata``.
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    sharded: Dict[str, object] = {}
    _load_state_dict(
        sharded,
        storage_reader=dcp.FileSystemReader(path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    checkpoint: Dict[str, object] = {"policy_state_dict": sharded.get("model", {})}
    meta_path = os.path.join(path, "metadata.pt")
    if os.path.exists(meta_path):
        checkpoint.update(torch.load(meta_path, map_location="cpu"))
    return checkpoint
