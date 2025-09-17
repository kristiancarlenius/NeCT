# nect/trainers/split_trainer.py
from __future__ import annotations
import json
from pathlib import Path
from typing import cast, Dict, Any

import torch
import torch.utils.data
from loguru import logger
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

from nect.trainers.base_trainer import BaseTrainer
from nect.data.encoded_dataset import EncodedShardDataset

MAX_POINTS_ENC_CHUNK = 5_000_000  # matches BaseTrainer comfort zone


def _collate_identity(batch):
    # batch is a list of dicts; for batch_size=1, return the dict directly
    assert len(batch) == 1, "We expect batch_size=1 for encoded shards"
    return batch[0]


class SplitTrainer(BaseTrainer):
    """
    Two-phase trainer for model='quadcubes_split':
      Phase A: run full_model.encode_inputs once over projector points -> save shards
      Phase B: train net only on saved encodings (no projector, no encoder)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # builds dataset, projector, full model, opts, schedulers, etc.

        # Sanity: must have split model (encoder + net)
        if not hasattr(self.model, "encoder") or not hasattr(self.model, "net"):
            raise ValueError("SplitTrainer requires a split model (encoder + net).")

        # keep a handle to the full model for inference/visualization/exports
        self.full_model = self.model

        # Encoded cache dir
        self.enc_dir = Path(self.checkpoint_directory_base).parent / "enc"
        if self.fabric.is_global_zero:
            self.enc_dir.mkdir(parents=True, exist_ok=True)
        self.fabric.barrier()

        manifest = self.enc_dir / "manifest.json"
        if not manifest.exists():
            logger.info("No encoded cache found — running pre-encoding on GPU…")
            self.preencode_all()
        else:
            logger.info("Found encoded cache — skipping pre-encoding.")

        # ---- switch to head-only training ----
        # Rebuild optimizer/schedulers for the MLP-only parameter set
        self.model = self.full_model.net  # share weights with full_model.net
        # Build a fresh optimizer for head-only
        self.optim = self.config.get_optimizer(self.model)
        (self.lr_scheduler_warmup,
         self.lr_scheduler,
         self.lr_scheduler_warmup_downsample) = self.config.get_lr_schedulers(self.optim)
        # Fabric wrap
        self.model, self.optim = self.fabric.setup(self.model, self.optim)

        # Encoded shard dataloader (one shard per step)
        ds = EncodedShardDataset(self.enc_dir)
        self.dataloader = cast(
            torch.utils.data.DataLoader,
            self.fabric.setup_dataloaders(
                torch.utils.data.DataLoader(
                    ds,
                    batch_size=1,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    collate_fn=_collate_identity,  # <- clean dict
                )
            ),
        )

    # -------------------------
    # Phase A: pre-encoding
    # -------------------------
    @torch.no_grad()
    def preencode_all(self) -> None:
        self.full_model.eval()
        manifest = []

        # Use the projection dataloader built by BaseTrainer (batch_size=1)
        for proj_idx, (proj, angle, timestep) in enumerate(self.dataloader):
            # NOTE: here self.dataloader is still the projection loader (we haven't swapped it yet)
            proj = proj.to(self.fabric.device, non_blocking=True).squeeze(0)
            angle = float(angle if not torch.is_tensor(angle) else angle.item())
            timestep = float(timestep if not torch.is_tensor(timestep) else timestep.item())

            # Maintain BaseTrainer’s scheduling / logging cadence
            self.on_angle_start(proj, angle)

            # Iterate batches of rays -> points via projector
            for batch_num in range(min(cast(int, self.batch_per_proj), self.projector.batch_per_epoch)):
                points, y = self.projector(batch_num=batch_num, proj=self.proj)
                if points is None or y is None:
                    continue

                # Flatten, filter invalid
                zero_mask = torch.all(points.view(-1, 3) == 0, dim=-1)  # (R*P,)
                pts_shape = points.size()  # (R, P, 3)
                if pts_shape[1] == 0:
                    self.logger.warning("No points in the batch")
                    continue

                flat_pts = points.view(-1, 3)[~zero_mask]
                if flat_pts.numel() == 0:
                    continue

                # Encode in chunks on GPU using the model's encode_inputs (NOT encoder.encode_inputs)
                encoded_chunks = []
                for s in range(0, flat_pts.size(0), MAX_POINTS_ENC_CHUNK):
                    sl = slice(s, s + MAX_POINTS_ENC_CHUNK)
                    enc = self.full_model.encode_inputs(flat_pts[sl].to(self.fabric.device), timestep)
                    encoded_chunks.append(enc.detach().to(dtype=torch.float16).cpu())
                X = torch.cat(encoded_chunks, dim=0)  # (N_valid, F)

                # Persist shard
                shard_name = f"shard_{proj_idx:05d}_{batch_num:04d}.pt"
                shard_path = self.enc_dir / shard_name
                bundle = {
                    "X": X,                                   # (N_valid, F) float16
                    "y": y.detach().cpu().float(),            # (R,)
                    "shape": (int(pts_shape[0]), int(pts_shape[1])),  # (R, P)
                    "mask": zero_mask.detach().cpu(),         # (R*P,) bool
                    "dist": float(self.projector.distances),  # scalar; if per-ray, save the tensor instead
                }
                torch.save(bundle, shard_path)
                manifest.append({"file": shard_name})

            self.on_angle_end()

        # Write manifest once
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

                tqdm_bar = self.tqdm(enumerate(self.dataloader),
                                     total=len(self.dataloader), leave=False, desc="Shards")

                for i, shard in tqdm_bar:
                    self.optim.zero_grad(set_to_none=True)
                    self.model.train()

                    X: torch.Tensor = shard["X"].to(self.fabric.device, non_blocking=True).to(dtype=torch.float16)
                    y: torch.Tensor = shard["y"].to(self.fabric.device, non_blocking=True).float()
                    mask: torch.Tensor = shard["mask"].to(self.fabric.device, non_blocking=True)
                    R, P = int(shard["shape"][0]), int(shard["shape"][1])
                    dist = shard["dist"]  # float; if tensor, move to device

                    # forward on points-only in chunks
                    preds_chunks = []
                    for s in range(0, X.size(0), MAX_POINTS_ENC_CHUNK):
                        sl = slice(s, s + MAX_POINTS_ENC_CHUNK)
                        preds = self.model(X[sl])                # (n, 1)
                        preds_chunks.append(preds.to(torch.float32))
                    atten_valid = torch.cat(preds_chunks, dim=0).view(-1, 1)  # (N_valid, 1)

                    # reinflate to (R*P,1), fill masked with 0
                    full = torch.zeros((R * P, 1), dtype=torch.float32, device=self.fabric.device)
                    full[~mask.view(-1)] = atten_valid.view(-1, 1)

                    # sum along the ray & scale by distance
                    atten_hat = full.view(R, P)
                    y_pred = torch.sum(atten_hat, dim=1) * (dist / (self.geometry.max_distance_traveled))

                    loss = self.loss_fn(y_pred, y)
                    self.fabric.backward(loss)
                    if self.config.clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
                    self.optim.step()
                    self.step += 1

                    mem = nvmlDeviceGetMemoryInfo(h)
                    tqdm_bar.set_postfix({
                        "loss": float(loss.item()),
                        "GPU": f"{round(mem.used/1024**3,1)}/{round(mem.total/1024**3,1)}G"
                    })

                self.on_train_epoch_end()

            self.evaluate()
            self.save_model(last=True)
        except KeyboardInterrupt:
            if self.step > 3000:
                self.logger.info("Please wait; saving model…")
                self.save_model(last=True)
            else:
                self.logger.info(f"Training ran {self.step} steps; not saving on early interrupt.")

    # -------- overrides to keep inference working (we swapped self.model to head-only) --------
    def create_volume(self, *args, **kwargs):
        # Use the full model (encoder + net) for volume export/preview
        orig_model = self.model
        try:
            self.model = self.full_model
            return super().create_volume(*args, **kwargs)
        finally:
            self.model = orig_model

    def save_model(self, last: bool = False):
        # Save the full model so you can load & reconstruct later
        if self.save or (last and self.save_last):
            state = {
                "model": self.full_model.state_dict(),  # encoder + net
                "epoch": self.current_epoch,
                "angle": self.current_angle,
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            if self.save_optimizer:
                state["optim"] = self.optim.state_dict()
            self.logger.info("Saving model - time might take some time")
            self.fabric.save(str(Path(self.checkpoint_directory_base) / "last.ckpt"), state)
            self.logger.info("Saving model finished")
            self.create_volume(self.config.save_volume)
            self.last_checkpoint_time = torch.cuda.Event(enable_timing=False)
        elif last:
            self.create_volume(self.config.save_volume)
