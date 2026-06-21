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
from nect.trainers.continous_scanning_trainer import ContinousScanningTrainer


class ContinousScanningTrainerBatch(ContinousScanningTrainer):
    """
    Continuous scanning trainer that collects all sub-angle points upfront,
    runs a single batched forward pass over all of them, then does a single backward.

    Compared to ContinousScanningTrainerMem:
    - Same number of forward passes (N) and backwards (1)
    - One large matrix multiply instead of N small ones → better GPU utilisation
    - Graph memory is the same order (total activations = N × points_per_sub_angle)

    Compared to ContinousScanningTrainer (2-pass):
    - Half the forward passes (N vs 2N), far fewer backwards (1 vs N)
    - Higher peak memory (all activations in one graph vs one sub-angle at a time)
    """

    def fit(self):
        self.step = 0
        self.training_time = time.perf_counter()
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        mem_info = nvmlDeviceGetMemoryInfo(h)
        total_gb = mem_info.total / 1024**3
        print(f"[GPU] Total memory: {total_gb:.1f} GB")

        for epoch in self.tqdm(range(self.current_epoch, self.config.epochs), total=self.config.epochs, leave=True, desc="Epochs"):
            self._epoch_loss_sum = 0.0
            self._epoch_loss_count = 0
            self.on_train_epoch_start()
            tqdm_bar = self.tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False, desc="Projections")

            for i, (proj, angle_start, angle_stop, timestep) in tqdm_bar:
                if i < self.current_angle:
                    continue

                self.on_angle_start(proj, angle_start)
                memory_info = nvmlDeviceGetMemoryInfo(h)
                if self.verbose:
                    tqdm_bar.set_postfix({"GPU mem%": f"{round(int(memory_info.used)/1024**3, 1)}/{int(memory_info.total)/1024**3}G"})
                    tqdm_bar.refresh()

                for batch_num in range(min(cast(int, self.batch_per_proj), self.projector.batch_per_epoch)):
                    self.optim.zero_grad()
                    self.model.train()

                    end_linspace = np.linspace(float(angle_start.detach().cpu()), float(angle_stop.detach().cpu()), self.config.accumulation_steps + 1, endpoint=True)
                    linspace = [(end_linspace[k] + end_linspace[k + 1]) / 2 for k in range(self.config.accumulation_steps)]
                    points_per_batch = 5000000

                    # Phase 1: collect points and metadata for all sub-angles without model calls.
                    # distances must be cloned since projector state is mutated each update_angle.
                    collected = []
                    for ang in linspace:
                        self.projector.update_angle(ang)
                        points, y = self.projector(batch_num=batch_num, proj=self.proj)
                        if points is None or y is None:
                            collected.append(None)
                            continue
                        zero_points_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
                        points_shape = points.size()
                        if points_shape[1] == 0:
                            self.logger.warning("No points in the batch")
                            collected.append(None)
                            continue
                        points_filtered = points.view(-1, 3)[~zero_points_mask]
                        if points_filtered.size(0) == 0:
                            collected.append(None)
                            continue
                        collected.append((points_filtered, zero_points_mask, points_shape, self.projector.distances.clone(), y))

                    valid = [(idx, c) for idx, c in enumerate(collected) if c is not None]
                    if not valid:
                        continue

                    # Phase 2: single batched forward pass over all sub-angle points.
                    all_points = torch.cat([c[0] for _, c in valid])
                    split_sizes = [c[0].size(0) for _, c in valid]

                    atten_hats = []
                    for p0 in range(0, all_points.size(0), points_per_batch):
                        chunk = all_points[p0:p0 + points_per_batch]
                        if self.config.mode == "dynamic":
                            atten_hat = self.model(chunk, float(timestep))
                        else:
                            atten_hat = self.model(chunk)
                        atten_hats.append(atten_hat)
                    all_atten_hat = torch.cat(atten_hats)

                    # Phase 3: split outputs back per sub-angle and accumulate y_pred / y_target.
                    split_outputs = torch.split(all_atten_hat, split_sizes)
                    y_pred = None
                    y_target = None

                    for (_, cached), atten_hat_sub in zip(valid, split_outputs):
                        points_filtered, zero_points_mask, points_shape, distances, y = cached
                        processed_tensor = torch.zeros((points_shape[0], points_shape[1], 1), dtype=torch.float32, device=self.fabric.device).view(-1, 1)
                        processed_tensor[~zero_points_mask] = atten_hat_sub
                        atten_hat_grid = processed_tensor.view(points_shape[0], points_shape[1])
                        contrib = torch.sum(atten_hat_grid, dim=1) * (distances / self.geometry.max_distance_traveled) / self.config.accumulation_steps

                        if y_pred is None:
                            y_pred = contrib
                            y_target = y / self.config.accumulation_steps
                        else:
                            y_pred = y_pred + contrib
                            y_target = y_target + y / self.config.accumulation_steps

                    if y_pred is None or y_target is None:
                        continue

                    if self.config.add_poisson:
                        y_pred_for_loss = (y_pred + torch.poisson(y_pred.detach() * 1e5) / 1e5) / 2
                    else:
                        y_pred_for_loss = y_pred

                    if self.config.s3im and self.current_projection > self.config.warmup.steps:
                        loss = 0
                        patch_size = min(math.floor(math.sqrt(self.projector.total_detector_pixels)), math.floor(math.sqrt(self.batch_size)))
                        self.fabric.log_dict({"patch_size": patch_size}, step=self.step)
                        loss += self.loss_fn(y_pred_for_loss, y_target)
                        loss += self.s3im_loss(y_pred_for_loss, y_target, patch_size=patch_size)
                    else:
                        loss = self.loss_fn(y_pred_for_loss, y_target)

                    self.fabric.backward(loss)

                    self.fabric.log_dict(
                        {
                            "loss": loss.item(),
                            "max_mem": torch.cuda.max_memory_allocated(),
                            "current_mem": int(memory_info.used),
                            "epoch": epoch,
                            "downsample_detector_factor": self.downsample_detector_factor,
                            "points_per_ray": self.points_per_ray,
                            "distance_between_points": self.projector.distance_between_points,
                            "lr": self.optim.param_groups[0]["lr"],
                            "num_proj_processed": self.current_projection,
                        },
                        step=self.step,
                    )
                    if hasattr(self.model, "skip_alpha"):
                        self.fabric.log_dict({"skip_alpha_value": self.model.skip_alpha.item()}, step=self.step)
                    if self.config.clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
                    self.optim.step()
                    self.step += 1
                    if torch.isfinite(loss):
                        self._epoch_loss_sum += float(loss.item())
                        self._epoch_loss_count += 1
                self.on_angle_end()
            self.on_train_epoch_end()
            torch.cuda.empty_cache()

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        mem_info = nvmlDeviceGetMemoryInfo(h)
        current_gb = mem_info.used / 1024**3
        print(f"[GPU] Peak allocated: {peak_gb:.2f} GB | Current used: {current_gb:.2f} GB / {total_gb:.1f} GB")
        self.evaluate()
        self.save_model(last=True)
