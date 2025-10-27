from __future__ import annotations
import json
from pathlib import Path
from typing import cast, Dict, Any
import logging
import torch
import numpy
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
        #steps, lr_multi = self.config.get_w0()
        #self.warmup_w0_only(steps=steps, lr_mult=lr_multi, include_b0=False)

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
    #add = 3 if (getattr(cfg.net, "include_identity", False) or False) else 0
    return L * F #+ add


def _encoded_width_quadcubes(cfg: Config) -> int:
    # QuadCubes output width: 4*(L*F) (+4 if include_identity)
    L = cfg.encoder.n_levels
    F = cfg.encoder.n_features_per_level
    #add = 4 if (getattr(cfg.net, "include_identity", False) or False) else 0
    return 4 * (L * F) #+ add

def _mlp_layer_splits(in_dim: int, net_cfg) -> list[int]:
    """
    Return per-layer param sizes in TCNN flat order:
      [W0, b0, W1, b1, ..., W_out, b_out]  (with output_dim = 1)

    Strategy:
      1) Compute the analytic sizes for FullyFusedMLP (no padding).
      2) Build a TCNN dummy (Identity encoder, same net cfg) and read the true
         flattened size from 'net.params'/'params'.
      3) Adjust ONLY the first layer's weight size (W0) by the difference so that
         sum(splits) == flat.numel(). This accounts for TCNN internal padding
         tied to D_in; later layers remain intact.
    """
    # Pull H and L & concrete dict for tcnn
    if hasattr(net_cfg, "n_neurons"):
        H = int(net_cfg.n_neurons)
        L = int(net_cfg.n_hidden_layers)
        net_conf = net_cfg.get_network_config()
    else:
        net_conf = net_cfg.get_network_config()
        H = int(net_conf["n_neurons"])
        L = int(net_conf["n_hidden_layers"])

    D_in = int(in_dim)
    D_out = 1  # single scalar output

    # Analytic (unpadded) splits
    splits: list[int] = []
    splits += [H * D_in, H]               # W0, b0
    for _ in range(L - 1):
        splits += [H * H, H]              # Wk, bk
    splits += [D_out * H, D_out]          # W_out, b_out

    # Validate vs dummy TCNN and fold any padding into W0
    enc = {"otype": "Identity", "n_dims_to_encode": D_in}
    dummy = tcnn.NetworkWithInputEncoding(
        n_input_dims=D_in,
        n_output_dims=D_out,
        encoding_config=enc,
        network_config=net_conf,
    )
    flat = dummy.state_dict().get("net.params", dummy.state_dict().get("params"))
    assert flat is not None, "TCNN dummy state_dict missing 'net.params'/'params'."

    diff = flat.numel() - sum(splits)
    if diff != 0:
        splits[0] += diff  # absorb padding into W0 only

    assert sum(splits) == flat.numel(), (
        f"Splits do not match TCNN storage even after padding W0: "
        f"sum={sum(splits)} vs tcnn={flat.numel()} (D_in={D_in}, H={H}, L={L})"
    )
    return splits


def _transfer_hashgrid_to_quadcubes(hg_sd: dict, qc_model: torch.nn.Module, hash_config_path: str | Path, qc_cfg: Config, logger=print, ) -> None:
    qc_sd = qc_model.state_dict()
    if "net.params" not in hg_sd or "net.params" not in qc_sd:
        logger("Checkpoint missing net.params")
        return

    hg_params = hg_sd["net.params"]
    qc_params = qc_sd["net.params"]

    logger(f"HashGrid net.params: {hg_params.shape}")
    logger(f"QuadCubes net.params: {qc_params.shape}")

    hg_cfg = get_cfg(hash_config_path, model="hash_grid", static=True)

    # Encoded input widths (no include_identity handling)
    hg_in = _encoded_width_hash(hg_cfg)
    qc_in = _encoded_width_quadcubes(qc_cfg)

    # Layer-wise MLP splits (padded to TCNN via dummy)
    hg_splits = _mlp_layer_splits(hg_in, hg_cfg.net)
    qc_splits = _mlp_layer_splits(qc_in, qc_cfg.net)

    # Encoder sizes
    enc_size_hg_total = hg_params.numel() - sum(hg_splits)
    enc_size_qc_total = qc_params.numel() - sum(qc_splits)
    enc0_size_qc = enc_size_qc_total // 4

    logger(f"HashGrid enc_size={enc_size_hg_total}, MLP size ={sum(hg_splits)}")
    logger(f"HashGrid MLP layers ={hg_splits}")
    logger(f"QuadCubes enc_total={enc_size_qc_total}, MLP sizes={sum(qc_splits)}")
    logger(f"QuadCubes MLP splits={qc_splits}")

    qc_new = qc_params.clone()
    damp_multi = qc_cfg.get_dm()
    scales = [damp_multi[0], damp_multi[1], damp_multi[1], damp_multi[1]]

    # ---- Encoder copy (checks only; no auto-fix) ----
    enc_src = hg_params[:enc_size_hg_total]
    ok_encoder = True
    if enc_size_qc_total % 4 != 0:
        logger(f"[ERR] QC encoder params ({enc_size_qc_total}) not divisible by 4.")
        ok_encoder = False
    if enc_src.numel() != enc0_size_qc:
        logger(f"[ERR] Encoder size mismatch: HG enc={enc_src.numel()} vs QC quarter={enc0_size_qc}")
        ok_encoder = False

    logger(f"Encoder check — HG enc={enc_src.numel()}, QC enc_total={enc_size_qc_total}, QC enc/4={enc0_size_qc}")

    if not ok_encoder:
        logger("[ABORT] Not copying encoders due to mismatch.")
    else:
        for i, s in enumerate(scales):
            lo = i * enc0_size_qc
            hi = (i + 1) * enc0_size_qc
            with torch.no_grad():
                qc_new[lo:hi] = enc_src * s
        logger("[OK] Encoders copied (4 quarters).")

    # ---- MLP copy (shape-accurate, padded-aware; checks only) ----
    hg_mlp = hg_params[enc_size_hg_total:]
    qc_mlp = qc_new[enc_size_qc_total:]

    def prefix_offsets(sizes: list[int]) -> list[int]:
        offs = [0]
        for s in sizes:
            offs.append(offs[-1] + s)
        return offs

    hg_off = prefix_offsets(hg_splits)
    qc_off = prefix_offsets(qc_splits)

    # Sizes from padded splits
    W0_hg_size = hg_splits[0]
    b0_hg_size = hg_splits[1]
    W0_qc_size = qc_splits[0]
    b0_qc_size = qc_splits[1]

    # Build slices we'll need regardless of branch
    W0_hg  = hg_mlp[hg_off[0]:hg_off[1]]
    W0_qc  = qc_mlp[qc_off[0]:qc_off[1]]
    b0_hg  = hg_mlp[hg_off[1]:hg_off[2]]
    b0_qc  = qc_mlp[qc_off[1]:qc_off[2]]
    tail_hg = hg_mlp[hg_off[2]:]
    tail_qc = qc_mlp[qc_off[2]:]

    ok_mlp = True
    if b0_hg_size != b0_qc_size:
        logger(f"[ERR] b0 size mismatch: HG={b0_hg_size} vs QC={b0_qc_size}")
        ok_mlp = False

    quarter = W0_qc_size // 4
    if W0_qc_size % 4 != 0:
        logger(f"[ERR] QC W0 ({W0_qc_size}) not divisible by 4.")
        ok_mlp = False
    if W0_hg_size != quarter:
        logger(f"[ERR] W0 mismatch: HG W0={W0_hg_size} vs QC quarter={quarter} (expected equal after padding).")
        ok_mlp = False

    if tail_hg.numel() != tail_qc.numel():
        logger(f"[ERR] Tail (layers 1..end) size mismatch: HG={tail_hg.numel()} vs QC={tail_qc.numel()}")
        ok_mlp = False

    logger(f"MLP check — W0: HG={W0_hg_size}, QC={W0_qc_size} (quarter={quarter}); b0: {b0_hg_size}; tail: HG={tail_hg.numel()}, QC={tail_qc.numel()}")

    # Decide what to copy
    only_tail_ok = (b0_hg_size == b0_qc_size) and (tail_hg.numel() == tail_qc.numel())

    if not ok_mlp and only_tail_ok:
        logger("[WARN] W0 layout differs; copying b0 and tail,only not anymore.")
        with torch.no_grad():
            W0_qc[:quarter] = W0_hg[:quarter] * damp_multi[2]
            W0_qc[quarter:] *= damp_multi[1]
            b0_qc[:] = b0_hg[:] * damp_multi[2]
            tail_qc[:] = tail_hg[:] * damp_multi[2]

    elif ok_mlp:
        # Full copy: W0 (tiled), b0, and tail
        W0_hg  = hg_mlp[hg_off[0]:hg_off[1]]
        W0_qc  = qc_mlp[qc_off[0]:qc_off[1]]
        with torch.no_grad():
            for i, s in enumerate(scales):
                lo = i * quarter
                hi = (i + 1) * quarter
                W0_qc[lo:hi] = W0_hg * s

            b0_qc[:] = b0_hg[:] * damp_multi[2]
            tail_qc[:] = tail_hg[:] * damp_multi[2]
            logger("[OK] MLP copied (W0 tiled into 4 quarters, b0 and tail copied).")
    else:
        logger("[ABORT] Not copying MLP.")

    qc_sd["net.params"] = qc_new
    missing, unexpected = qc_model.load_state_dict(qc_sd, strict=False)
    logger(f"Final load: Missing={len(missing)}, Unexpected={len(unexpected)}")
    #logger(f"The start network:{qc_sd.detach().cpu().numpy()}")



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

