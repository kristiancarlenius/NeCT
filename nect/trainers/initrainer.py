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
    Return list of param sizes per MLP layer (weights + bias) in TCNN packing order
    [W0, b0, W1, b1, ..., W_out, b_out], for output_dim=1.

    We compute splits analytically from the network config (reliable for FullyFusedMLP),
    and then *validate* against a dummy NetworkWithInputEncoding to ensure the total
    param count matches TCNN's actual flattened size.
    """
    # Pull H and L from either the object or its dict
    if hasattr(net_cfg, "n_neurons"):
        H = int(net_cfg.n_neurons)
        L = int(net_cfg.n_hidden_layers)
        net_conf = net_cfg.get_network_config()
    else:
        net_conf = net_cfg.get_network_config()
        H = int(net_conf["n_neurons"])
        L = int(net_conf["n_hidden_layers"])

    D_in = int(in_dim)
    D_out = 1

    # Analytic per-layer splits (sizes in flattened 1D storage)
    splits: list[int] = []
    # input -> hidden0
    splits += [H * D_in, H]
    # hidden stacks: (L-1) transitions of HxH + H
    for _ in range(L - 1):
        splits += [H * H, H]
    # hidden_last -> output
    splits += [D_out * H, D_out]

    # Validate against a dummy net's flat params length
    enc = {"otype": "Identity", "n_dims_to_encode": D_in}
    dummy = tcnn.NetworkWithInputEncoding(
        n_input_dims=D_in,
        n_output_dims=D_out,
        encoding_config=enc,
        network_config=net_conf,
    )
    flat = dummy.state_dict().get("net.params", dummy.state_dict().get("params"))
    assert flat is not None, "TCNN dummy state_dict missing 'net.params'/'params'."
    assert sum(splits) == flat.numel(), (
        f"MLP split mismatch: computed={sum(splits)} vs tcnn={flat.numel()} "
        f"(D_in={D_in}, H={H}, L={L})"
    )
    return splits


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

    # Encoded input widths
    hg_in = _encoded_width_hash(hg_cfg)
    qc_in = _encoded_width_quadcubes(qc_cfg)

    # Layer-wise MLP splits (validated by dummy)
    hg_splits = _mlp_layer_splits(hg_in, hg_cfg.net)
    qc_splits = _mlp_layer_splits(qc_in, qc_cfg.net)

    # Encoder param sizes (total - MLP total)
    enc_size_hg_total = hg_params.numel() - sum(hg_splits)
    enc_size_qc_total = qc_params.numel() - sum(qc_splits)
    enc0_size_qc = enc_size_qc_total // 4

    logger(f"HashGrid enc_size={enc_size_hg_total}, MLP splits={hg_splits}")
    logger(f"QuadCubes enc_total={enc_size_qc_total}, enc0={enc0_size_qc}, MLP splits={qc_splits}")

    qc_new = qc_params.clone()
    scales = [0.8, 0.4, 0.4, 0.4]

    # ---- Encoder copy (require exact match; otherwise just report) ----
    enc_src = hg_params[:enc_size_hg_total]
    ok_encoder = True
    if enc_size_qc_total % 4 != 0:
        logger(f"[ERR] QC encoder params ({enc_size_qc_total}) not divisible by 4.")
        ok_encoder = False
    if enc_src.numel() != enc0_size_qc:
        logger(f"[ERR] Encoder size mismatch: HG enc={enc_src.numel()} vs QC quarter={enc0_size_qc}")
        ok_encoder = False

    logger(
        f"Encoder check — HG enc={enc_src.numel()}, "
        f"QC enc_total={enc_size_qc_total}, QC enc/4={enc0_size_qc}"
    )

    if not ok_encoder:
        logger("[ABORT] Not copying encoders due to mismatch.")
    else:
        for i, s in enumerate(scales):
            lo = i * enc0_size_qc
            hi = (i + 1) * enc0_size_qc
            qc_new[lo:hi] = enc_src * s
        logger("[OK] Encoders copied (4 quarters).")

    # ---- MLP copy (shape-accurate, checks only; no silent fixes) ----
    hg_mlp = hg_params[enc_size_hg_total:]
    qc_mlp = qc_new[enc_size_qc_total:]

    def prefix_offsets(sizes: list[int]) -> list[int]:
        offs = [0]
        for s in sizes:
            offs.append(offs[-1] + s)
        return offs

    hg_off = prefix_offsets(hg_splits)
    qc_off = prefix_offsets(qc_splits)

    W0_hg_size = hg_splits[0]
    b0_hg_size = hg_splits[1]
    W0_qc_size = qc_splits[0]
    b0_qc_size = qc_splits[1]

    ok_mlp = True
    if b0_hg_size != b0_qc_size:
        logger(f"[ERR] b0 size mismatch: HG={b0_hg_size} vs QC={b0_qc_size}")
        ok_mlp = False
    if W0_qc_size % 4 != 0:
        logger(f"[ERR] QC W0 ({W0_qc_size}) not divisible by 4.")
        ok_mlp = False

    quarter = W0_qc_size // 4
    if W0_hg_size != quarter:
        logger(f"[ERR] W0 mismatch: HG W0={W0_hg_size} vs QC quarter={quarter} (expected equal).")
        ok_mlp = False

    tail_hg = hg_mlp[hg_off[2]:]
    tail_qc = qc_mlp[qc_off[2]:]
    if tail_hg.numel() != tail_qc.numel():
        logger(f"[ERR] Tail (layers 1..end) size mismatch: HG={tail_hg.numel()} vs QC={tail_qc.numel()}")
        ok_mlp = False

    logger(
        f"MLP check — W0: HG={W0_hg_size}, QC={W0_qc_size} (quarter={quarter}); "
        f"b0: {b0_hg_size}; tail: HG={tail_hg.numel()}, QC={tail_qc.numel()}"
    )

    if not ok_mlp:
        logger("[ABORT] Not copying MLP due to mismatch.")
    else:
        # Proceed to copy since all checks passed
        W0_hg  = hg_mlp[hg_off[0]:hg_off[1]]
        b0_hg  = hg_mlp[hg_off[1]:hg_off[2]]
        W0_qc  = qc_mlp[qc_off[0]:qc_off[1]]
        b0_qc  = qc_mlp[qc_off[1]:qc_off[2]]

        # Tile W0 into 4 quarters; copy b0 and the rest
        for i, s in enumerate(scales):
            lo = i * quarter
            hi = (i + 1) * quarter
            W0_qc[lo:hi] = W0_hg * s

        b0_qc[:] = b0_hg * 0.4
        tail_qc[:] = tail_hg[:] * 0.4
        logger("[OK] MLP copied (W0 tiled into 4 quarters, b0 and tail copied).")

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


    """
    # ---- Encoder copy (four equal blocks) ----
    enc_src = hg_params[:enc_size_hg_total]
    enc0 = enc_size_qc_total // 4
    for i, s in enumerate(scales):
        lo = i * enc0
        qc_new[lo:lo+enc0] = enc_src * s

    logger(f"Should be equal hg, qc/4: {enc_src.numel(), enc0}")

    # ---- Encoder copy ----
    enc_src = hg_params[:enc_size_hg_total]
    enc_dst = qc_new[:enc0_size_qc]
    n_enc_copy = min(enc_src.numel(), enc_dst.numel())
    
    enc_dst[:n_enc_copy] = enc_src[:n_enc_copy]*0.8

    logger(f"Copied encoder0: {n_enc_copy}/{enc_dst.numel()}")

    #Adding into the other encoders 
    damp_mult = 0.4
    enc_dst[n_enc_copy:n_enc_copy*2] = enc_src[:n_enc_copy]*damp_mult    #*(1/6) #1/4*2/3
    enc_dst[n_enc_copy*2:n_enc_copy*3] = enc_src[:n_enc_copy]*damp_mult
    enc_dst[n_enc_copy*3:n_enc_copy*4] = enc_src[:n_enc_copy]*damp_mult

    logger(f"Copied to the other three endocers (from, to, to, to): {n_enc_copy, n_enc_copy*2, n_enc_copy*3, n_enc_copy*4}")
    logger(f"Total should be equal to 4x encoder (tot, 4*enc): {enc_size_qc_total, n_enc_copy*4}")
    
    # ---- MLP copy (shape-accurate) ----
    hg_mlp = hg_params[enc_size_hg_total:]
    qc_mlp = qc_new[enc_size_qc_total:]

    def prefix_offsets(sizes):
        offs = [0]
        for s in sizes:
            offs.append(offs[-1] + s)
        return offs

    hg_off = prefix_offsets(hg_splits)
    qc_off = prefix_offsets(qc_splits)

    # W0 / b0 sizes via splits
    W0_hg_size = hg_splits[0]
    b0_size    = hg_splits[1]              # should equal qc_splits[1]
    W0_qc_size = qc_splits[0]

    # Sanity checks
    logger(f"Hidden width (b0) must match: {b0_size, qc_splits[1]}")
    logger(f"QC W0 must be divisible by 4 (True): {W0_qc_size % 4 == 0}")
    logger(f"HG W0 must be equal (or less?) one QC quarter: {W0_hg_size, (W0_qc_size // 4)}")

    # Offsets inside the MLP tails
    W0_hg_lo, W0_hg_hi = hg_off[0], hg_off[1]        # [0 : W0_hg_size]
    b0_hg_lo, b0_hg_hi = hg_off[1], hg_off[2]        # [W0_hg : W0_hg + b0]
    tail_hg_lo         = hg_off[2]                    # start of W1

    W0_qc_lo, W0_qc_hi = qc_off[0], qc_off[1]
    b0_qc_lo, b0_qc_hi = qc_off[1], qc_off[2]
    tail_qc_lo         = qc_off[2]

    # Slice out the actual flat segments
    W0_hg = hg_mlp[W0_hg_lo:W0_hg_hi]
    b0_hg = hg_mlp[b0_hg_lo:b0_hg_hi]
    tail_hg = hg_mlp[tail_hg_lo:]          # (W1,b1,...) flattened

    W0_qc = qc_mlp[W0_qc_lo:W0_qc_hi]
    b0_qc = qc_mlp[b0_qc_lo:b0_qc_hi]
    tail_qc = qc_mlp[tail_qc_lo:]

    # Tile HG's W0 into the four QC quarters
    quarter = W0_qc_size // 4
    copy_len = min(W0_hg_size, quarter)

    # scales for the four branches (first gets the strongest)
    for i, s in enumerate(scales):
        lo = i * quarter
        hi = lo + copy_len
        W0_qc[lo:hi] = W0_hg[:copy_len] * s
        # zero any leftover space in the quarter (optional)
        if quarter > copy_len:
            logger("Size differences in input weights")

    # Copy b0 directly (same hidden width)
    b0_qc[:] = b0_hg[:] * 0.4

    # Copy the rest of the layers 1..end directly (same shapes)
    logger(f"Hidden/topology beyond input layer qc should be same as hg: {tail_qc.numel(), tail_hg.numel()}")
    tail_qc[:] = tail_hg[:] * 0.4

    # ---- MLP copy ----
    hg_mlp = hg_params[enc_size_hg_total:]
    qc_mlp = qc_new[enc_size_qc_total:]
    logger(f"HashGrid MLP ={hg_mlp.size()}")
    logger(f"Quadcubes MLP ={qc_mlp.size()}")

    mlp_overflow = qc_mlp.size()-hg_mlp.size()
    encoder_input_layer_size = int(mlp_overflow/3)

    qc_mlp[:encoder_input_layer_size] = hg_mlp[:encoder_input_layer_size]*damp_mult
    qc_mlp[encoder_input_layer_size:encoder_input_layer_size*2] = hg_mlp[:encoder_input_layer_size]*damp_mult
    qc_mlp[encoder_input_layer_size*2:encoder_input_layer_size*3] = hg_mlp[:encoder_input_layer_size]*damp_mult
    qc_mlp[encoder_input_layer_size*3:encoder_input_layer_size*4] = hg_mlp[:encoder_input_layer_size]*damp_mult
    qc_mlp[encoder_input_layer_size*4:] = hg_mlp[encoder_input_layer_size:]*damp_mult

    logger(f"Copied over nn and copies of input layer: {encoder_input_layer_size}")
    logger(f"Remaining network should be equal for qc and hc: {qc_mlp.size()-encoder_input_layer_size*4, hg_mlp.size()-encoder_input_layer_size}")

    
    logger(f"HashGrid MLP no-input-layer={hg_mlp.size()}")
    logger(f"Quadcubes MLP no-input-layer={qc_mlp.size()}")
    # cumulative offsets
    hg_offsets = [0] + list(torch.cumsum(torch.tensor(hg_splits), dim=0).tolist())
    qc_offsets = [0] + list(torch.cumsum(torch.tensor(qc_splits), dim=0).tolist())

    # skip layer 0 (input layer)
    for li in range(1, len(hg_splits)):  # start from hidden layer 1
        hg_slice = hg_mlp[hg_offsets[li]:hg_offsets[li+1]]
        qc_slice = qc_mlp[qc_offsets[li]:qc_offsets[li+1]]
        n_copy = min(hg_slice.numel(), qc_slice.numel())
        qc_slice[:n_copy] = hg_slice[:n_copy]*0.3
        logger(f"Copied MLP layer {li}: {n_copy}/{qc_slice.numel()}")
    
    """