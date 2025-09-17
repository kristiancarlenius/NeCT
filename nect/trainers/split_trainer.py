# nect/trainers/split_trainer.py
from __future__ import annotations
import json, math, os, time
from pathlib import Path
from typing import cast, Dict, Any

import numpy as np
import torch
import torch.utils.data
from loguru import logger
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

from nect.trainers.base_trainer import BaseTrainer
from nect.data.encoded_dataset import EncodedShardDataset

# hard cap like BaseTrainer (tcnn comfort zone)
MAX_POINTS_ENC_CHUNK = 5_000_000

class SplitTrainer(BaseTrainer):
    """
    Two-phase trainer for model='quadcubes_split':
      Phase A: run model.encoder once over projector points -> save shards
      Phase B: train self.model.net on saved encodings (no projector, no encoder)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # builds dataset, projector, model, etc.
        # Sanity: must have split model
        if not hasattr(self.model, "encoder") or not hasattr(self.model, "net"):
            raise ValueError("SplitTrainer requires a split model (encoder + net).")

        # Where to store encoded shards
        self.enc_dir = Path(self.checkpoint_directory_base).parent / "enc"
        if self.fabric.is_global_zero:
            self.enc_dir.mkdir(parents=True, exist_ok=True)
        self.fabric.barrier()

        # Phase A (encode once) only if no cache exists
        manifest = self.enc_dir / "manifest.json"
        if not manifest.exists():
            logger.info("No encoded cache found — running pre-encoding on GPU…")
            self.preencode_all()
        else:
            logger.info("Found existing encoded cache — skipping pre-encoding.")

        # After pre-encoding, discard encoder and keep only MLP
        self.model = self.model.net  # only the MLP from now on
        self.model, self.optim = self.fabric.setup(self.model, self.optim)

        # Build a dataloader over encoded shards
        ds = EncodedShardDataset(self.enc_dir)
        # One shard per step (shards can be big); shuffle=True for better mixing
        self.dataloader = cast(
            torch.utils.data.DataLoader,
            self.fabric.setup_dataloaders(
                torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=self.config.num_workers)
            ),
        )

    # -------------------------
    # Phase A: pre-encoding
    # -------------------------
    @torch.no_grad()
    def preencode_all(self) -> None:
        self.model.eval()
        # keep encoder on device, mixed-precision ok (Fabric set to 16-mixed)
        manifest = []
        shard_id = 0

        for proj_idx, (proj, angle, timestep) in enumerate(self.dataloader):
            # BaseTrainer created dataloader with batch_size=1; unwrap
            proj = proj.to(self.fabric.device, non_blocking=True).squeeze(0)
            timestep = float(timestep.item() if torch.is_tensor(timestep) else timestep)

            # Let BaseTrainer update internal state (downsampling, points_per_ray, etc.)
            self.on_angle_start(proj, float(angle))

            # Iterate batches of rays -> points via projector
            for batch_num in range(min(cast(int, self.batch_per_proj), self.projector.batch_per_epoch)):
                points, y = self.projector(batch_num=batch_num, proj=self.proj)
                if points is None or y is None:
                    continue

                zero_mask = torch.all(points.view(-1, 3) == 0, dim=-1)  # (R*P,)
                pts_shape = points.size()  # (R, P, 3)
                if pts_shape[1] == 0:
                    self.logger.warning("No points in the batch")
                    continue

                flat_pts = points.view(-1, 3)[~zero_mask]
                if flat_pts.numel() == 0:
                    continue

                # Encode in chunks on GPU
                encoded_chunks = []
                for s in range(0, flat_pts.size(0), MAX_POINTS_ENC_CHUNK):
                    sl = slice(s, s + MAX_POINTS_ENC_CHUNK)
                    enc = self.model.encoder.encode_inputs(flat_pts[sl].to(self.fabric.device), timestep)
                    encoded_chunks.append(enc.detach().to(dtype=torch.float16).cpu())  # fast + small
                X = torch.cat(encoded_chunks, dim=0)  # (N_valid, F)

                # Persist shard
                shard_name = f"shard_{proj_idx:05d}_{batch_num:04d}.pt"
                shard_path = self.enc_dir / shard_name
                bundle = {
                    "X": X,                                   # features
                    "y": y.detach().cpu().float(),            # (R,)
                    "shape": (int(pts_shape[0]), int(pts_shape[1])),  # (R, P)
                    "mask": zero_mask.detach().cpu(),         # (R*P,)
                    "dist": float(self.projector.distances),  # scalar
                }
                torch.save(bundle, shard_path)
                manifest.append({"file": shard_name})

            self.on_angle_end()

        # write manifest
        if self.fabric.is_global_zero:
            with open(self.enc_dir / "manifest.json", "w") as f:
                json.dump(manifest, f)
        self.fabric.barrier()
        logger.info(f"Pre-encoding complete. Saved {len(manifest)} shards to {self.enc_dir}")

    # -------------------------
    # Phase B: train MLP on encodings
    # -------------------------
    def fit(self):
        try:
            self.step = 0
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)

            for epoch in self.tqdm(range(self.current_epoch, self.config.epochs),
                                   total=self.config.epochs, initial=self.current_epoch, desc="Epochs"):
                self.on_train_epoch_start()
                tqdm_bar = self.tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False, desc="Shards")

                for i, batch in tqdm_bar:
                    self.optim.zero_grad(set_to_none=True)
                    self.model.train()

                    # Each dataset item is a shard dict (batch_size=1). Unwrap & move
                    shard: Dict[str, Any] = batch  # DataLoader(batch_size=1) returns dict of tensors wrapped in lists
                    # When batch_size=1, DataLoader returns each tensor with leading dim==1; handle both cases:
                    X = shard["X"][0] if isinstance(shard["X"], list) or (isinstance(shard["X"], torch.Tensor) and shard["X"].dim()==3) else shard["X"]
                    y = shard["y"][0] if isinstance(shard["y"], list) or (isinstance(shard["y"], torch.Tensor) and shard["y"].dim()==2) else shard["y"]
                    mask = shard["mask"][0] if isinstance(shard["mask"], list) or (isinstance(shard["mask"], torch.Tensor) and shard["mask"].dim()==2) else shard["mask"]
                    shape = shard["shape"][0] if isinstance(shard["shape"], list) else shard["shape"]
                    dist = shard["dist"][0].item() if isinstance(shard["dist"], list) else shard["dist"]

                    R, P = int(shape[0]), int(shape[1])
                    X = X.to(self.fabric.device).to(dtype=torch.float16)  # as saved
                    y = y.to(self.fabric.device).float()
                    mask = mask.to(self.fabric.device)

                    # forward in chunks (points-only), then reassemble to rays
                    preds_chunks = []
                    for s in range(0, X.size(0), MAX_POINTS_ENC_CHUNK):
                        sl = slice(s, s + MAX_POINTS_ENC_CHUNK)
                        preds = self.model(X[sl])  # (n, 1)
                        preds_chunks.append(preds.to(dtype=torch.float32))  # sum in fp32
                    atten_valid = torch.cat(preds_chunks, dim=0).view(-1, 1)  # (N_valid, 1)

                    # build full [R*P, 1] with zeros at masked positions
                    full = torch.zeros((R * P, 1), dtype=torch.float32, device=self.fabric.device)
                    full[~mask.view(-1)] = atten_valid.view(-1, 1)

                    # reshape -> sum along points to get detector prediction
                    atten_hat = full.view(R, P)
                    y_pred = torch.sum(atten_hat, dim=1) * (dist / (self.geometry.max_distance_traveled))

                    # loss & step
                    loss = self.loss_fn(y_pred, y)
                    self.fabric.backward(loss)
                    if self.config.clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
                    self.optim.step()
                    self.step += 1

                    # logs
                    mem = nvmlDeviceGetMemoryInfo(h)
                    tqdm_bar.set_postfix({"loss": float(loss.item()),
                                          "GPU": f"{round(mem.used/1024**3,1)}/{round(mem.total/1024**3,1)}G"})

                self.on_train_epoch_end()

            self.evaluate()
            self.save_model(last=True)
        except KeyboardInterrupt:
            if self.step > 3000:
                self.logger.info("Please wait; saving model…")
                self.save_model(last=True)
            else:
                self.logger.info(f"Training ran {self.step} steps; not saving on early interrupt.")
