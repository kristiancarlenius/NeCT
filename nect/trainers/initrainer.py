from __future__ import annotations
import json
from pathlib import Path
from typing import cast, Dict, Any
import logging
import torch
import torch.utils.data
from loguru import logger
from nect.trainers.base_trainer import BaseTrainer
from nect.config import setup_cfg, load_config
from typing import Literal, Optional
MAX_POINTS_ENC_CHUNK = 5_000_000  # matches BaseTrainer comfort zone


class IniTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        output_directory=None,
        checkpoint: Optional[str] = None,
        static_init: Optional[str] = None,
        static_init_config: Optional[str] = None,  # NEW: yaml config path for HashGrid
        init_mode: Literal["hash_to_quadcubes", "direct"] = "hash_to_quadcubes",
        **kwargs,
    ):
        super().__init__(config=config, output_directory=output_directory, checkpoint=checkpoint, **kwargs)

        if static_init is None:
            return

        self.logger.info(
            f"Initializing model from '{static_init}' "
            f"with mode '{init_mode}'"
        )

        ckpt = torch.load(static_init, map_location="cpu")
        sd = _extract_state_dict(ckpt)

        if init_mode == "direct":
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            self.logger.info(f"Loaded directly. Missing={len(missing)}, Unexpected={len(unexpected)}")
        elif init_mode == "hash_to_quadcubes":
            if static_init_config is None:
                raise ValueError("static_init_config (yaml for HashGrid) must be provided for hash_to_quadcubes mode")
            _transfer_hashgrid_to_quadcubes(sd, self.model, static_init_config, logger=self.logger.info)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    return ckpt_obj


def _transfer_hashgrid_to_quadcubes(hg_sd: dict, qc_model: torch.nn.Module, hash_config_path: str, logger=print) -> None:
    """Copy HashGrid → QuadCubes: encoder (xyz) exact, MLP partial with padding."""

    qc_sd = qc_model.state_dict()
    if "net.params" not in hg_sd or "net.params" not in qc_sd:
        logger("Checkpoint missing net.params")
        return

    hg_params = hg_sd["net.params"]
    qc_params = qc_sd["net.params"]

    logger(f"HashGrid net.params: {hg_params.shape}")
    logger(f"QuadCubes net.params: {qc_params.shape}")

    # Query offsets from live QuadCubes
    qc_offsets = qc_model.net.offsets.cpu().numpy()
    logger(f"QuadCubes offsets: {qc_offsets}")

    # Build dummy HashGrid model using its original config
    from nect.network import HashGrid
    hg_config = setup_cfg(load_config(hash_config_path))
    dummy_hg = HashGrid(
        encoding_config=hg_config.encoder,
        network_config=hg_config.net,
    )
    hg_offsets = dummy_hg.net.offsets.cpu().numpy()
    logger(f"HashGrid offsets (from dummy model): {hg_offsets}")

    # Slice
    hg_enc_slice = slice(hg_offsets[0], hg_offsets[1])
    hg_mlp_slice = slice(hg_offsets[1], hg_offsets[-1])

    qc_enc0_slice = slice(qc_offsets[0], qc_offsets[1])
    qc_mlp_slice = slice(qc_offsets[-2], qc_offsets[-1])

    logger(f"Encoder slice HashGrid={hg_enc_slice}, QuadCubes={qc_enc0_slice}")
    logger(f"MLP slice HashGrid={hg_mlp_slice}, QuadCubes={qc_mlp_slice}")

    qc_params_new = qc_params.clone()

    # ---- Encoder copy ----
    enc_src = hg_params[hg_enc_slice]
    enc_dst = qc_params_new[qc_enc0_slice]
    if enc_src.shape == enc_dst.shape:
        qc_params_new[qc_enc0_slice] = enc_src
        logger(f"Copied encoder (x,y,z): {enc_src.shape}")
    else:
        logger(f"Encoder mismatch: src={enc_src.shape}, dst={enc_dst.shape}")

    # ---- MLP copy with padding ----
    mlp_src = hg_params[hg_mlp_slice]
    mlp_dst = qc_params_new[qc_mlp_slice]

    Ncopy = min(mlp_src.numel(), mlp_dst.numel())
    qc_params_new[qc_mlp_slice][:Ncopy] = mlp_src[:Ncopy]
    logger(f"Copied MLP {Ncopy}/{mlp_src.numel()} values into {mlp_dst.numel()}")

    # Replace in state dict
    qc_sd["net.params"] = qc_params_new
    missing, unexpected = qc_model.load_state_dict(qc_sd, strict=False)
    logger(f"Final load: Missing={len(missing)}, Unexpected={len(unexpected)}")