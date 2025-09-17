from __future__ import annotations

import math
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.utils.data
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

from nect.config import Config
from nect.trainers.base_trainer import BaseTrainer

torch.autograd.set_detect_anomaly(True)

class SplitTrainer(BaseTrainer):
    def __init__(
        self,
        config: Config,
        output_directory: str | Path | None = None,
        checkpoint: str | Path | None = None,
        save_ckpt: bool = True,
        save_last: bool = True,
        save_optimizer: bool = True,
        verbose: bool = True,
        log: bool = True,
        cancel_at: str | None = None,
        keep_two: bool = True,
        prune: bool = True,
        preencode: bool = True,   # <--- new flag
    ):
        super().__init__(
            config=config,
            output_directory=output_directory,
            checkpoint=checkpoint,
            save_ckpt=save_ckpt,
            save_last=save_last,
            save_optimizer=save_optimizer,
            verbose=verbose,
            log=log,
            cancel_at=cancel_at,
            keep_two=keep_two,
            prune=prune,
        )
        self.preencode = preencode
        if not hasattr(self.model, "encoder") or not hasattr(self.model, "net"):
            raise ValueError("SplitTrainer requires QuadCubesSplit (with encoder + net).")

        if self.preencode:
            self.logger.info("Precomputing encodings for dataset...")
            self.precompute_encodings()
            # Replace model with MLP only
            self.model = self.model.net

    def precompute_encodings(self):
        all_encoded = []
        all_targets = []
        for proj, angle, timestep in self.dataloader.dataset:
            # Build point cloud as usual
            points, y = self.projector(batch_num=0, proj=proj)
            if points is None:
                continue
            zero_points_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
            points = points.view(-1, 3)[~zero_points_mask]
            if points.numel() == 0:
                continue
            encoded = self.model.encoder.encode_inputs(points.to(self.fabric.device), float(timestep))
            all_encoded.append(encoded.detach().cpu())
            all_targets.append(y.detach().cpu())

        self.encoded_dataset = torch.utils.data.TensorDataset(
            torch.cat(all_encoded, dim=0),
            torch.cat(all_targets, dim=0),
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.encoded_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def fit(self):
        self.step = 0
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        for epoch in self.tqdm(range(self.current_epoch, self.config.epochs),
                               total=self.config.epochs, desc="Epochs"):
            self.on_train_epoch_start()
            tqdm_bar = self.tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False, desc="Batches")
            for i, (encoded, y) in tqdm_bar:
                self.optim.zero_grad()
                self.model.train()
                y_pred = self.model(encoded.to(self.fabric.device)).squeeze(-1)
                loss = self.loss_fn(y_pred, y.to(self.fabric.device))
                self.fabric.backward(loss)
                if self.config.clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
                self.optim.step()
                self.step += 1
                memory_info = nvmlDeviceGetMemoryInfo(h)
                tqdm_bar.set_postfix({
                    "loss": float(loss.item()),
                    "GPU mem": f"{round(memory_info.used/1024**3,1)}/{round(memory_info.total/1024**3,1)}G"
                })
            self.on_train_epoch_end()
        self.evaluate()
        self.save_model(last=True)
