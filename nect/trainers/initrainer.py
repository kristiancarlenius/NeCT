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
    qc_sd = qc_model.state_dict()
    if "net.params" not in hg_sd or "net.params" not in qc_sd:
        logger("Checkpoint missing net.params")
        return

    hg_params = hg_sd["net.params"]
    qc_params = qc_sd["net.params"]

    logger(f"HashGrid net.params: {hg_params.shape}")
    logger(f"QuadCubes net.params: {qc_params.shape}")

    # Build dummy HashGrid to know split sizes
    from nect.network import HashGrid
    hg_config = setup_cfg(load_config(hash_config_path))
    dummy_hg = HashGrid(
        encoding_config=hg_config.encoder,
        network_config=hg_config.net,
    )

    n_enc_hg = dummy_hg.net.encoding.n_params()
    n_net_hg = dummy_hg.net.network.n_params()
    logger(f"HashGrid split: enc={n_enc_hg}, mlp={n_net_hg}, total={n_enc_hg+n_net_hg}")

    n_enc0_qc = qc_model.net.encoding.nested[0].n_params()
    n_net_qc = qc_model.net.network.n_params()
    logger(f"QuadCubes split: enc0={n_enc0_qc}, mlp={n_net_qc}, total={n_enc0_qc+n_net_qc}")

    qc_params_new = qc_params.clone()

    # ---- Copy encoder ----
    enc_src = hg_params[:n_enc_hg]
    enc_dst = qc_params_new[:n_enc0_qc]
    if enc_src.shape == enc_dst.shape:
        qc_params_new[:n_enc0_qc] = enc_src
        logger(f"Copied encoder: {enc_src.shape}")
    else:
        logger(f"Encoder mismatch: src={enc_src.shape}, dst={enc_dst.shape}")

    # ---- Copy MLP with padding ----
    mlp_src = hg_params[n_enc_hg:]
    mlp_dst = qc_params_new[-n_net_qc:]

    Ncopy = min(mlp_src.numel(), mlp_dst.numel())
    mlp_dst[:Ncopy] = mlp_src[:Ncopy]
    logger(f"Copied MLP {Ncopy}/{mlp_src.numel()} into {mlp_dst.numel()}")

    qc_sd["net.params"] = qc_params_new
    missing, unexpected = qc_model.load_state_dict(qc_sd, strict=False)
    logger(f"Final load: Missing={len(missing)}, Unexpected={len(unexpected)}")




#from nect.network import HashGrid
#hg_config = setup_cfg(load_config(hash_config_path))