from __future__ import annotations
import json
from pathlib import Path
from typing import cast, Dict, Any
import logging
import torch
import torch.utils.data
from loguru import logger
from nect.trainers.base_trainer import BaseTrainer
from typing import Literal, Optional
MAX_POINTS_ENC_CHUNK = 5_000_000  # matches BaseTrainer comfort zone

class IniTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        output_directory=None,
        checkpoint: Optional[str] = None,   # resume same-arch
        static_init: Optional[str] = None,  # preload from HashGrid or same-arch weights
        init_mode: Literal["hash_to_quadcubes","direct"] = "hash_to_quadcubes",
        **kwargs,
    ):
        super().__init__(config=config, output_directory=output_directory, checkpoint=checkpoint, **kwargs)

        # If we actually resumed training (checkpoint=...), skip init-from-static
        if static_init is None:
            return

        self.logger.info(f"Initializing model from '{static_init}' with mode '{init_mode}'")
        sd = _extract_state_dict_any(static_init)
        qc_sd = self.model.state_dict()

        print("\n=== HashGrid ckpt keys (first 20) ===")
        for k in list(sd.keys())[:20]:
            print(k, sd[k].shape)

        print("\n=== QuadCubes model keys (first 30) ===")
        for k in list(qc_sd.keys())[:30]:
            print(k, qc_sd[k].shape)

        if init_mode == "direct":
            # Use *only* when architectures match exactly.
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            self.logger.info(f"Direct load complete. Missing={len(missing)}, Unexpected={len(unexpected)}")
            return

        if init_mode == "hash_to_quadcubes":
            qc_sd = self.model.state_dict()
            remapped = _remap_hashgrid_to_quadcubes(sd, qc_sd, log=self.logger.info)

            # Merge remapped into the current QC state and load
            qc_sd.update(remapped)
            missing, unexpected = self.model.load_state_dict(qc_sd, strict=False)

            self.logger.info(f"Transferred tensors: {len(remapped)}")
            # Show a few actual mappings for sanity:
            for i, (k, _) in enumerate(remapped.items()):
                if i >= 8: break
                self.logger.info(f"  copied → {k}")
            # These 'missing' will typically be the 3 time-encoders
            self.logger.info(f"Missing={len(missing)} (expected extra encoders), Unexpected={len(unexpected)}")
            return

        raise ValueError(f"Unknown init_mode: {init_mode}")

def _extract_state_dict_any(ckpt_path: str) -> dict:
    """Return a pure state_dict from a variety of checkpoint formats."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # epoch-style: explicit dict of tensors
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]

    # sometimes people save raw state_dict
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    # fabric 'last.ckpt' often stores the model object
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_obj = ckpt["model"]
        if hasattr(model_obj, "state_dict") and callable(model_obj.state_dict):
            return model_obj.state_dict()
        if isinstance(model_obj, dict):  # rare, but handle anyway
            return model_obj

    raise ValueError(f"Could not extract a state_dict from: {ckpt_path}")

def _remap_hashgrid_to_quadcubes(hg_sd: dict, qc_sd: dict, log=print) -> dict:
    """
    Build a *remapped* state_dict containing:
      - encoder(x,y,z): HashGrid 'net.encoding.*'  -> QuadCubes 'net.encoding.nested.0.*'
      - MLP:            HashGrid 'net.network.*'   -> QuadCubes 'net.network.*'
    Only keys that exist & match shape on the QuadCubes side are kept.
    """
    remapped = {}
    transferred_enc = transferred_mlp = 0

    # (A) Encoder: HashGrid has single encoding at 'net.encoding.*'
    # Build target as 'net.encoding.nested.0.' + tail
    for k, v in hg_sd.items():
        if not k.startswith("net.encoding."):
            continue
        tail = k[len("net.encoding."):]         # e.g. 'params' or 'levels.0.params' etc.
        tgt = f"net.encoding.nested.0.{tail}"   # encoder 0 is (x,y,z) branch
        if tgt in qc_sd and qc_sd[tgt].shape == v.shape:
            remapped[tgt] = v
            transferred_enc += 1

    # (B) MLP: names usually match 1:1
    for k, v in hg_sd.items():
        if not k.startswith("net.network."):
            continue
        if k in qc_sd and qc_sd[k].shape == v.shape:
            remapped[k] = v
            transferred_mlp += 1

    log(f"Encoder(x,y,z) tensors prepared: {transferred_enc}")
    log(f"MLP tensors prepared: {transferred_mlp}")
    return remapped

