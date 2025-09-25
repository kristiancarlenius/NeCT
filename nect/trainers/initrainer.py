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
from nect.config import get_cfg
from nect.network import HashGrid, QuadCubes

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
        print("Hash_grid sanity check")
        sanity_check_params(static_init_config, "hash_grid")
        print("Quadcubes sanity check")
        sanity_check_params("/cluster/home/kristiac/NeCT/outputs/dynamic_initilalized/model/config.yaml", "quadcubes")

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

    # Load full config (so SAME_FOLDER resolves geometry)
    hg_config = get_cfg(hash_config_path, model="hash_grid", static=True)

    # ---- Compute HashGrid parameter split ----
    n_levels = hg_config.encoder.n_levels
    n_feat = hg_config.encoder.n_features_per_level
    log2_size = hg_config.encoder.log2_hashmap_size

    table_size = (1 << log2_size)
    per_level_size = n_feat * table_size
    enc_size_hg = n_levels * per_level_size

    # MLP params = total - encoder
    mlp_size_hg = hg_params.numel() - enc_size_hg

    # ---- Compute QuadCubes encoder0 size ----
    qc_n_levels = hg_config.encoder.n_levels
    qc_feat = hg_config.encoder.n_features_per_level
    qc_log2_size = hg_config.encoder.log2_hashmap_size

    qc_table_size = (1 << qc_log2_size)
    per_level_size_qc = qc_feat * qc_table_size
    enc_size_qc = qc_n_levels * per_level_size_qc

    # For MLP in QuadCubes: everything else
    mlp_size_qc = qc_params.numel() - 4 * enc_size_qc

    logger(f"Split HashGrid: enc={enc_size_hg}, mlp={mlp_size_hg}")
    logger(f"Split QuadCubes: enc0={enc_size_qc}, mlp={mlp_size_qc}")

    qc_params_new = qc_params.clone()

    # ---- Copy encoder ----
    enc_src = hg_params[:enc_size_hg]
    enc_dst = qc_params_new[:enc_size_qc]
    Ncopy = min(enc_src.numel(), enc_dst.numel())
    enc_dst[:Ncopy] = enc_src[:Ncopy]
    logger(f"Copied encoder0: {Ncopy}/{enc_dst.numel()} values")

    # ---- Copy MLP ----
    mlp_src = hg_params[enc_size_hg:]
    mlp_dst = qc_params_new[-mlp_size_qc:]
    Ncopy = min(mlp_src.numel(), mlp_dst.numel())
    mlp_dst[:Ncopy] = mlp_src[:Ncopy]
    logger(f"Copied MLP: {Ncopy}/{mlp_dst.numel()} values")

    qc_sd["net.params"] = qc_params_new
    missing, unexpected = qc_model.load_state_dict(qc_sd, strict=False)
    logger(f"Final load: Missing={len(missing)}, Unexpected={len(unexpected)}")

def sanity_check_params(config_path: str, model: str = "hash_grid"):
    """
    Build model from config and print a breakdown of net.params into encoder/MLP.
    This helps verify param ordering in tiny-cuda-nn checkpoints.
    """
    cfg = get_cfg(config_path, model=model, static=(model == "hash_grid"))
    net = cfg.get_model()
    sd = net.state_dict()

    if "net.params" not in sd:
        print(f"[{model}] no net.params in state_dict!")
        return

    params = sd["net.params"]
    print(f"[{model}] net.params total: {params.numel()} ({params.shape})")

    if model == "hash_grid":
        # derive encoder param count
        n_levels = cfg.encoder.n_levels
        n_feat = cfg.encoder.n_features_per_level
        table_size = 1 << cfg.encoder.log2_hashmap_size
        enc_size = n_levels * n_feat * table_size
        mlp_size = params.numel() - enc_size
        print(f"[{model}] expect enc_size={enc_size}, mlp_size={mlp_size}")
        print(f"  encoder slice: params[0:{enc_size}]")
        print(f"  mlp slice: params[{enc_size}:]")

    elif model == "quadcubes":
        n_levels = cfg.encoder.n_levels
        n_feat = cfg.encoder.n_features_per_level
        table_size = 1 << cfg.encoder.log2_hashmap_size
        enc_size = n_levels * n_feat * table_size
        total_enc = 4 * enc_size
        mlp_size = params.numel() - total_enc
        print(f"[{model}] expect enc0_size={enc_size}, total_enc={total_enc}, mlp_size={mlp_size}")
        print(f"  encoder0 slice: params[0:{enc_size}]")
        print(f"  encoder1 slice: params[{enc_size}:{2*enc_size}]")
        print(f"  encoder2 slice: params[{2*enc_size}:{3*enc_size}]")
        print(f"  encoder3 slice: params[{3*enc_size}:{4*enc_size}]")
        print(f"  mlp slice: params[{4*enc_size}:]")

    else:
        print(f"Sanity checker not implemented for model={model}")



#from nect.network import HashGrid
#hg_config = setup_cfg(load_config(hash_config_path))