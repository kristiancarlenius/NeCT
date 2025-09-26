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
from nect.config import Config, get_cfg
from nect.network import HashGrid, QuadCubes
import tinycudann as tcnn

MAX_POINTS_ENC_CHUNK = 5_000_000  # matches BaseTrainer comfort zone


class IniTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        output_directory=None,
        checkpoint: Optional[str] = None,
        static_init: Optional[str] = None,
        static_init_config: Optional[str] = None,
        init_mode: str = "hash_to_quadcubes",
        **kwargs,
    ):
        super().__init__(config=config, output_directory=output_directory, checkpoint=checkpoint, **kwargs)

        if not static_init or init_mode != "hash_to_quadcubes":
            return

        
        self.logger.info(f"Initializing model from '{static_init}' with mode '{init_mode}'")
        ckpt = torch.load(static_init, map_location="cpu")
        sd = ckpt["model"] if "model" in ckpt else ckpt

        if not static_init_config:
            raise ValueError("static_init_config (path to saved static HashGrid config.yaml) is required")

        _transfer_hashgrid_to_quadcubes(sd, self.model, hash_config_path=static_init_config, qc_cfg=self.config, logger=self.logger.info)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    return ckpt_obj

def _estimate_mlp_params_via_identity(in_dim: int, net_cfg) -> int:
    """
    Build a tiny NetworkWithInputEncoding that uses Identity(in_dim) as the encoder
    (0 params) and the same MLP config. Then the size of net.params is exactly the
    MLP parameter count (including all tcnn padding/alignment).
    """
    enc = {"otype": "Identity", "n_dims_to_encode": int(in_dim)}
    dummy = tcnn.NetworkWithInputEncoding(
        n_input_dims=in_dim,
        n_output_dims=1,
        encoding_config=enc,
        network_config=net_cfg.get_network_config(),
    )
    return dummy.state_dict()["net.params"].numel()


def _encoded_width_hash(cfg: Config) -> int:
    # HashGrid output width: L * F (+3 if include_identity)
    L = cfg.encoder.n_levels
    F = cfg.encoder.n_features_per_level
    add = 3 if (getattr(cfg.net, "include_identity", False) or False) else 0
    return L * F + add


def _encoded_width_quadcubes(cfg: Config) -> int:
    # QuadCubes output width: 4*(L*F) (+4 if include_identity)
    L = cfg.encoder.n_levels
    F = cfg.encoder.n_features_per_level
    add = 4 if (getattr(cfg.net, "include_identity", False) or False) else 0
    return 4 * (L * F) + add


def _transfer_hashgrid_to_quadcubes(
    hg_sd: dict,
    qc_model: torch.nn.Module,
    hash_config_path: str | Path,
    qc_cfg: Config,
    logger=print,
    ) -> None:
    """
    Copy HashGrid → QuadCubes (encoder0 + MLP) using exact MLP sizes measured via Identity encoders.
    - No use of non-existent tcnn attributes.
    - Works with FullyFusedMLP padding/alignment.
    """
    qc_sd = qc_model.state_dict()
    if "net.params" not in hg_sd or "net.params" not in qc_sd:
        logger("Checkpoint missing net.params")
        return

    hg_params = hg_sd["net.params"]
    qc_params = qc_sd["net.params"]

    logger(f"HashGrid net.params: {hg_params.shape}")
    logger(f"QuadCubes net.params: {qc_params.shape}")

    # Load the saved static config (resolves SAME_FOLDER -> geometry.yaml etc.)
    hg_cfg = get_cfg(hash_config_path, model="hash_grid", static=True)
    #sanity_check_params_exact(hg_cfg, "hash_grid", logger)
    #sanity_check_params_exact(qc_cfg, "quadcubes", logger)

    # ---- Exact MLP sizes via Identity encoders ----
    hg_in = _encoded_width_hash(hg_cfg)
    qc_in = _encoded_width_quadcubes(qc_cfg)

    mlp_size_hg = _estimate_mlp_params_via_identity(hg_in, hg_cfg.net)
    mlp_size_qc = _estimate_mlp_params_via_identity(qc_in, qc_cfg.net)
    
    # Encoders are "everything else"
    enc_size_hg_total = hg_params.numel() - mlp_size_hg
    enc_size_qc_total = qc_params.numel() - mlp_size_qc
    if enc_size_qc_total % 4 != 0:
        logger(f"[warn] QuadCubes encoder total ({enc_size_qc_total}) not divisible by 4; rounding down.")
    enc0_size_qc = enc_size_qc_total // 4

    logger(f"Split HashGrid: mlp={mlp_size_hg}, enc={enc_size_hg_total}")
    logger(f"Split QuadCubes: mlp={mlp_size_qc}, enc_total={enc_size_qc_total}, enc0={enc0_size_qc}")

    # ---- Copy slices ----
    qc_new = qc_params.clone()

    # Encoder0: fill as much as possible from the HG encoder block
    enc_src = hg_params[:enc_size_hg_total]
    enc_dst = qc_new[:enc0_size_qc]
    n_enc_copy = min(enc_src.numel(), enc_dst.numel())
    enc_dst[:n_enc_copy] = enc_src[:n_enc_copy]
    logger(f"Copied encoder0: {n_enc_copy}/{enc_dst.numel()} values")

    # MLP: copy tail-to-tail (sizes can differ; copy the overlap)
    mlp_src = hg_params[-mlp_size_hg:] if mlp_size_hg > 0 else hg_params.new_empty(0)
    mlp_dst = qc_new[-mlp_size_qc:] if mlp_size_qc > 0 else qc_new.new_empty(0)
    n_mlp_copy = min(mlp_src.numel(), mlp_dst.numel())
    if n_mlp_copy > 0:
        mlp_dst[:n_mlp_copy] = mlp_src[:n_mlp_copy]
    logger(f"Copied MLP: {n_mlp_copy}/{mlp_dst.numel()} values")

    qc_sd["net.params"] = qc_new
    missing, unexpected = qc_model.load_state_dict(qc_sd, strict=False)
    logger(f"Final load: Missing={len(missing)}, Unexpected={len(unexpected)}")

def sanity_check_params_exact(cfg: Config, model_name: str, logger=print):
    if model_name == "hash_grid":
        in_dim = _encoded_width_hash(cfg)
    elif model_name == "quadcubes":
        in_dim = _encoded_width_quadcubes(cfg)
    else:
        logger(f"Unsupported model: {model_name}")
        return

    # Build a dummy same-MLP network but with Identity encoder to read exact MLP size
    mlp_exact = _estimate_mlp_params_via_identity(in_dim, cfg.net)

    # Build the real model and read total
    net = cfg.get_model()
    total = net.state_dict()["net.params"].numel()

    if model_name == "quadcubes":
        enc_total = total - mlp_exact
        logger(f"[{model_name}] total={total}, enc_total={enc_total}, enc0≈{enc_total//4}, mlp_exact={mlp_exact}")
    else:
        enc_total = total - mlp_exact
        logger(f"[{model_name}] total={total}, enc={enc_total}, mlp_exact={mlp_exact}")