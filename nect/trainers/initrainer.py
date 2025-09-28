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


def _estimate_mlp_params_via_identity(in_dim: int, net_cfg) -> int:
    enc = {"otype": "Identity", "n_dims_to_encode": int(in_dim)}
    dummy = tcnn.NetworkWithInputEncoding(
        n_input_dims=in_dim,
        n_output_dims=1,
        encoding_config=enc,
        network_config=net_cfg.get_network_config(),
    )
    sd = dummy.state_dict()
    print("Dummy state_dict keys:", sd.keys())
    if "net.params" in sd:
        return sd["net.params"].numel()
    elif "params" in sd:
        return sd["params"].numel()
    else:
        raise KeyError(f"No params key found in dummy net state_dict: {list(sd.keys())}")


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

def _mlp_layer_splits(in_dim: int, net_cfg) -> list[int]:
    """
    Return list of param sizes per MLP layer (weights + bias),
    using Identity encoder with in_dim to simulate tcnn layout.
    """
    enc = {"otype": "Identity", "n_dims_to_encode": int(in_dim)}
    dummy = tcnn.NetworkWithInputEncoding(
        n_input_dims=in_dim,
        n_output_dims=1,
        encoding_config=enc,
        network_config=net_cfg.get_network_config(),
    )
    params = dummy.state_dict().get("net.params", dummy.state_dict().get("params"))
    sizes = []
    offset = 0
    # Each layer (weight + bias) is stored sequentially in net.params
    for p in dummy.parameters():
        n = p.numel()
        sizes.append(n)
        offset += n
    assert sum(sizes) == params.numel()
    return sizes


def _transfer_hashgrid_to_quadcubes(
    hg_sd: dict,
    qc_model: torch.nn.Module,
    hash_config_path: str | Path,
    qc_cfg: Config,
    logger=print,
) -> None:
    qc_sd = qc_model.state_dict()
    if "net.params" not in hg_sd or "net.params" not in qc_sd:
        logger("Checkpoint missing net.params")
        return

    hg_params = hg_sd["net.params"]
    qc_params = qc_sd["net.params"]

    logger(f"HashGrid net.params: {hg_params.shape}")
    logger(f"QuadCubes net.params: {qc_params.shape}")

    hg_cfg = get_cfg(hash_config_path, model="hash_grid", static=True)

    # Enc sizes
    hg_in = _encoded_width_hash(hg_cfg)
    qc_in = _encoded_width_quadcubes(qc_cfg)

    # Layer-wise splits
    hg_splits = _mlp_layer_splits(hg_in, hg_cfg.net)
    qc_splits = _mlp_layer_splits(qc_in, qc_cfg.net)

    # Encoder param sizes
    enc_size_hg_total = hg_params.numel() - sum(hg_splits)
    enc_size_qc_total = qc_params.numel() - sum(qc_splits)
    enc0_size_qc = enc_size_qc_total // 4

    logger(f"HashGrid enc_size={enc_size_hg_total}, MLP splits={hg_splits}")
    logger(f"QuadCubes enc_total={enc_size_qc_total}, enc0={enc0_size_qc}, MLP splits={qc_splits}")

    qc_new = qc_params.clone()

    # ---- Encoder copy ----
    enc_src = hg_params[:enc_size_hg_total]
    enc_dst = qc_new[:enc0_size_qc]
    n_enc_copy = min(enc_src.numel(), enc_dst.numel())
    enc_dst[:n_enc_copy] = enc_src[:n_enc_copy]
    logger(f"Copied encoder0: {n_enc_copy}/{enc_dst.numel()}")

    # ---- MLP copy (skip first layer) ----
    hg_mlp = hg_params[enc_size_hg_total:]
    qc_mlp = qc_new[enc_size_qc_total:]

    # cumulative offsets
    hg_offsets = [0] + list(torch.cumsum(torch.tensor(hg_splits), dim=0).tolist())
    qc_offsets = [0] + list(torch.cumsum(torch.tensor(qc_splits), dim=0).tolist())

    # skip layer 0 (input layer)
    for li in range(1, len(hg_splits)):  # start from hidden layer 1
        hg_slice = hg_mlp[hg_offsets[li]:hg_offsets[li+1]]
        qc_slice = qc_mlp[qc_offsets[li]:qc_offsets[li+1]]
        n_copy = min(hg_slice.numel(), qc_slice.numel())
        qc_slice[:n_copy] = hg_slice[:n_copy]
        logger(f"Copied MLP layer {li}: {n_copy}/{qc_slice.numel()}")

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