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
        checkpoint: Optional[str] = None,          # normal resume (same-arch)
        static_init: Optional[str] = None,        # NEW: path to weights you want to initialize from
        init_mode: Literal["hash_to_quadcubes", "direct"] = "hash_to_quadcubes",
        **kwargs,
    ):
        """
        init_mode:
          - "hash_to_quadcubes": treat static_init as a trained HashGrid,
                                 transfer encoder(x,y,z)+MLP into QuadCubes
          - "direct": load_state_dict directly (same-architecture)
        """
        # Run normal BaseTrainer init (builds model, wraps with Fabric, etc.)
        super().__init__(config=config, output_directory=output_directory, checkpoint=checkpoint, **kwargs)

        if static_init is None:
            return  # nothing to do

        self.logger.info(f"Initializing model from '{static_init}' with mode '{init_mode}'")
        ckpt = torch.load(static_init, map_location="cpu")

        if init_mode == "direct":
            # Same-architecture init
            sd = _extract_state_dict(ckpt)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            self.logger.info(f"Loaded directly. Missing={len(missing)}, Unexpected={len(unexpected)}")

        elif init_mode == "hash_to_quadcubes":
            _transfer_hash_to_quadcubes(ckpt, self.model, logger=self.logger.info)

        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")
        

def _extract_state_dict(ckpt_obj):
    # Support {"model_state_dict": ...}, {"model": dict}, or raw dict
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    return ckpt_obj


def _transfer_hash_to_quadcubes(hash_ckpt, qc_model, logger=print):
    """
    Copy encoder(x,y,z) + MLP from a trained HashGrid into QuadCubes.
    Leaves the other encoders untouched.
    """
    qc_sd = qc_model.state_dict()
    hg_params = _extract_state_dict(hash_ckpt)["net.params"]  # flat vector

    # Query param sizes from tinycudann
    try:
        qc_enc0_size = qc_model.net.n_encoding_params
        qc_mlp_size = qc_model.net.n_network_params
    except Exception as e:
        logger(f"Failed to query n_encoding_params / n_network_params: {e}")
        return

    logger(f"=== HashGrid ckpt params ===")
    logger(f"HG flat tensor size: {hg_params.numel()}")
    logger(f"=== QuadCubes model params ===")
    logger(f"QC flat tensor size: {qc_sd['net.params'].numel()}")
    logger(f"QC encoder0 size: {qc_enc0_size}")
    logger(f"QC MLP size: {qc_mlp_size}")

    # Slice from HashGrid
    hg_enc = hg_params[:qc_enc0_size]
    hg_mlp = hg_params[-qc_mlp_size:]
    hg_middle = hg_params[qc_enc0_size: -qc_mlp_size]

    logger(f"Slicing offsets:")
    logger(f"  enc0 [: {qc_enc0_size}] → {hg_enc.numel()} params")
    logger(f"  middle [{qc_enc0_size} : -{qc_mlp_size}] → {hg_middle.numel()} params (unused)")
    logger(f"  mlp [-{qc_mlp_size}:] → {hg_mlp.numel()} params")

    # Prepare a new flat tensor for QuadCubes
    qc_params = qc_sd["net.params"].clone()

    # Replace encoder0 and MLP
    qc_params[:qc_enc0_size] = hg_enc
    qc_params[-qc_mlp_size:] = hg_mlp

    # Load back into state_dict
    qc_sd["net.params"] = qc_params
    qc_model.load_state_dict(qc_sd, strict=False)

    logger(f"Transferred encoder0 ({hg_enc.numel()} params) and MLP ({hg_mlp.numel()} params) "
           f"from HashGrid → QuadCubes")
