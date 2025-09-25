# nect/trainers/split_trainer.py
from __future__ import annotations
import json
from pathlib import Path
from typing import cast, Dict, Any

import torch
import torch.utils.data
from loguru import logger
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

from nect.trainers.base_trainer import BaseTrainer
from typing import Literal, Optional
MAX_POINTS_ENC_CHUNK = 5_000_000  # matches BaseTrainer comfort zone

class IniTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        output_directory = None,
        checkpoint: Optional[str] = None,          # normal resume (same-arch)
        static_init: Optional[str] = None,      # NEW: path to weights you want to initialize from
        init_mode: Literal["hash_to_quadcubes", "direct"] = "hash_to_quadcubes",
        **kwargs,
    ):
        """
        init_mode:
          - "hash_to_quadcubes": treat static_init as a trained HashGrid, transfer encoder(x,y,z)+MLP into QuadCubes
          - "direct": load_state_dict directly (same-architecture)
        """
        # Run the normal BaseTrainer init (builds model, wraps with Fabric, etc.)
        super().__init__(config=config, output_directory=output_directory, checkpoint=checkpoint, **kwargs)

        if static_init is None:
            return  # nothing to do

        # Load checkpoint (CPU is fine; tensors will be moved on load_state_dict)
        self.logger.info(f"Initializing model from '{static_init}' with mode '{init_mode}'")
        ckpt = torch.load(static_init, map_location="cpu")
        sd = _extract_state_dict(ckpt)

        if init_mode == "static":
            # Same-architecture init
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            self.logger.info(f"Loaded directly. Missing={len(missing)}, Unexpected={len(unexpected)}")
        elif init_mode == "hash_to_quadcubes":
            # Transfer HashGrid → QuadCubes
            _transfer_hashgrid_to_quadcubes(sd, self.model, logger=self.logger.info)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

def _extract_state_dict(ckpt_obj):
    # Support both {"model_state_dict": ...} and {"model": state_dict} or raw state_dict
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    # raw state_dict or something unexpected but dict-like
    return ckpt_obj

def _transfer_hashgrid_to_quadcubes(hg_sd: dict, qc_model: torch.nn.Module, logger=print) -> None:
    """
    Copy HashGrid weights into QuadCubes:
      - encoder (x,y,z) -> QuadCubes' first nested encoder
      - MLP -> MLP
    Leaves other encoders untouched.
    Falls back to shape-matching if key names differ.
    """
    qc_sd = qc_model.state_dict()
    transferred, skipped = [], []

    # 1) Heuristic direct mapping (most common with tcnn):
    #    HashGrid keys usually start with "net.encoding" (encoder) and "net.network" (MLP).
    #    QuadCubes composite encoder puts (x,y,z) under "net.encoding.nested.0"
    for k, v in hg_sd.items():
        if k.startswith("net.encoding"):
            tgt = k.replace("net.encoding", "net.encoding.nested.0")
        else:
            tgt = k  # MLP keys often identical: "net.network.*"
        if tgt in qc_sd and qc_sd[tgt].shape == v.shape:
            qc_sd[tgt] = v
            transferred.append((k, tgt))
        else:
            skipped.append(k)

    # 2) Shape-based fallback for any leftover keys (be conservative).
    #    Build a shape->list-of-keys map for the QuadCubes side.
    shape_to_qkeys = {}
    for qk, qv in qc_sd.items():
        shape_to_qkeys.setdefault(tuple(qv.shape), []).append(qk)

    for k in skipped[:]:
        v = hg_sd[k]
        shp = tuple(v.shape)
        if shp in shape_to_qkeys and shape_to_qkeys[shp]:
            tgt = shape_to_qkeys[shp].pop(0)
            # Avoid clobbering non-(encoder0/MLP) parts if names already matched:
            # only fallback if it looks like an encoder or MLP tensor
            if ("net.encoding" in k or "net.network" in k):
                qc_sd[tgt] = v
                transferred.append((k, tgt))
                skipped.remove(k)

    qc_model.load_state_dict(qc_sd, strict=False)

    logger(f"Transferred {len(transferred)} tensors from HashGrid → QuadCubes.")
    for s, t in transferred[:12]:
        logger(f"  {s}  →  {t}")
    if skipped:
        logger(f"Skipped {len(skipped)} tensors (no safe target).")
