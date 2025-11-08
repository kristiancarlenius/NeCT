from __future__ import annotations

import math
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.utils.data
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
import matplotlib.pyplot as plt
from nect.config import Config
from nect.trainers.base_trainer import BaseTrainer

torch.autograd.set_detect_anomaly(True)


class ContinousScanningTrainer(BaseTrainer):
    def __init__(
        self,
        config: Config,
        output_directory: str | Path | None = None,
        checkpoint: str | Path | None = None,
        **kwargs,
        #save_ckpt: bool = True,
        #save_last: bool = True,
        #save_optimizer: bool = True,
        #verbose: bool = True,
        #log: bool = True,
        #cancel_at: str | None = None,
        #keep_two: bool = True,
        #prune: bool = True,
    ):
        super().__init__(
            config=config,
            output_directory=output_directory,
            checkpoint=checkpoint,
            **kwargs,
            #save_ckpt=save_ckpt,
            #save_last=save_last,
            #save_optimizer=save_optimizer,
            #verbose=verbose,
            #log=log,
            #cancel_at=cancel_at,
            #keep_two=keep_two,
            #prune=prune
        )
        if config.accumulation_steps is None:
            raise ValueError("accumulation_steps must be provided")
        if config.continous_scanning is False:
            raise ValueError("continous_scanning must be True")

    def generate_image(self, prior: bool = False):
        with torch.no_grad():
            if self.config.points_per_batch == "auto":
                return
            
            plot = self.config.plot_type
            if plot is None:
                return
            
            if self.fabric.is_global_zero:
                size = [*self.config.geometry.nVoxel]
                sample_size = [*size]
                rm = self.config.sample_outside
                if rm > 0:
                    sample_size = [size[0], size[1] + 2 * rm, size[2] + 2 * rm]

                if plot == "XZ":
                    size[1] = 1

                elif plot == "YZ":
                    size[2] = 1

                elif plot == "XY":
                    size[0] = 1

                if size[0] * size[1] * size[2] > self.config.points_per_batch:
                    sample_size = [sample_size[i] // 3 for i in range(3)]
                    sample_size = [s if s > 0 else 1 for s in sample_size]

                z, y, x = torch.meshgrid(
                    [
                        torch.linspace(0, 1, steps=sample_size[0]) if plot != "XY" else torch.tensor(0.5),
                        torch.linspace(0, 1, steps=sample_size[1])[slice(rm, -rm) if rm > 0 else slice(None)] if plot != "XZ" else torch.tensor(0.5),
                        torch.linspace(0, 1, steps=sample_size[2])[slice(rm, -rm) if rm > 0 else slice(None)] if plot != "YZ" else torch.tensor(0.5),
                    ],
                    indexing="ij",
                )
                grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t().to(self.fabric.device)
                self.model.eval()
                if self.config.mode == "dynamic":
                    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
                    avg = self.model(grid, torch.tensor(0)).squeeze().reshape(size).squeeze().detach().cpu().numpy()
                    for i in range(3):
                        dynamic = (self.model(grid, torch.tensor((i + 1) / 4)).squeeze().reshape(size).squeeze().detach().cpu().numpy())
                        axes[0, i].imshow(dynamic - avg, cmap="gray", interpolation="none")
                        dynamic = dynamic / (self.geometry.max_distance_traveled * 2)
                        dynamic = dynamic * (self.dataset.maximum.item() - self.dataset.minimum.item())
                        dynamic = dynamic + self.dataset.minimum.item()
                        axes[1, i].imshow(dynamic, cmap="gray", interpolation="none")

                    for ax in axes.ravel():
                        ax.set_axis_off()

                    fig.tight_layout()
                else:
                    if size[0] * size[1] * size[2] < self.config.points_per_batch:
                        output = self.model(grid).squeeze().reshape(size).squeeze().detach().cpu().numpy()
                    else:
                        output = torch.zeros(size).numpy()
                        for i in range(size[0]):
                            output[i] = (self.model(grid[i * size[1] * size[2] : (i + 1) * size[1] * size[2]]).squeeze(0).reshape(size[1], size[2]).squeeze(1).detach().cpu().numpy())

                    output = output / self.geometry.max_distance_traveled
                    output = output * (self.dataset.maximum.item() - self.dataset.minimum.item())
                    output = output + self.dataset.minimum.item()
                    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
                    vmin = float(self.dataset.minimum.item())
                    vmax = float(0.75)
                    axes[0].hist(output.flatten(), bins=100, range=(vmin, vmax))
                    axes[1].imshow(output, cmap="gray", interpolation="none", vmin=vmin, vmax=vmax)
                save_path = f"{self.image_directory_base}/{self.current_epoch:04}_{self.current_angle:04}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()
            self.last_image_time = time.perf_counter()

    def fit(self):
        self.step = 0
        self.training_time = time.perf_counter()
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        for epoch in self.tqdm(range(self.current_epoch, self.config.epochs), total=self.config.epochs, leave=True, desc="Epochs",):
            self._epoch_loss_sum = 0.0
            self._epoch_loss_count = 0
            self.on_train_epoch_start()
            tqdm_bar = self.tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False, desc="Projections",)
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
                    y_pred = None
                    end_linspace = np.linspace(float(angle_start.detach().cpu()), float(angle_stop.detach().cpu()), self.config.accumulation_steps + 1, endpoint=True, )
                    linspace = [(end_linspace[k] + end_linspace[k + 1]) / 2 for k in range(self.config.accumulation_steps)]
                    for ang in linspace:
                        self.projector.update_angle(ang)
                        points, y = self.projector(batch_num=batch_num, proj=self.proj)
                        if points is None or y is None:
                            continue

                        zero_points_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
                        points_shape = points.size()
                        if points_shape[1] == 0:
                            self.logger.warning("No points in the batch")
                            continue

                        points = points.view(-1, 3)[~zero_points_mask]
                        if points.size(0) == 0:
                            continue

                        atten_hats = []
                        points_per_batch = 5000000  # 5 million points per batch is about the maximum that can be processed at once with tinycudann
                        for points_num in range(0, points.size(0), points_per_batch):
                            if self.config.mode == "dynamic":
                                atten_hat = self.model(points[points_num : points_num + points_per_batch], float(timestep),).squeeze(0)  # .view((points.size(0), points.size(1)))
                            else:
                                atten_hat = self.model(points[points_num : points_num + points_per_batch]).squeeze(0)  # .view((points.size(0), points.size(1)))
                            
                            atten_hats.append(atten_hat)

                        atten_hat = torch.cat(atten_hats)

                        processed_tensor = torch.zeros((points_shape[0], points_shape[1], 1), dtype=torch.float32, device=self.fabric.device,).view(-1, 1)
                        processed_tensor[~zero_points_mask] = atten_hat
                        atten_hat = processed_tensor.view(points_shape[0], points_shape[1])
                        if y_pred is None:
                            y_pred = (torch.sum(atten_hat, dim=1) * (self.projector.distances / (self.geometry.max_distance_traveled)) / self.config.accumulation_steps)  # * (self.ct_sampler.distance_between_points / self.geometry.max_distance_traveled)
                        else:
                            y_pred += (torch.sum(atten_hat, dim=1) * (self.projector.distances / (self.geometry.max_distance_traveled)) / self.config.accumulation_steps)
                            
                    if self.config.add_poisson:
                        y_pred = (y_pred + torch.poisson(y_pred * 1e5) / 1e5) / 2

                    if self.config.s3im and self.current_projection > self.config.warmup.steps:
                        loss = 0
                        patch_size = min(math.floor(math.sqrt(self.projector.total_detector_pixels)), math.floor(math.sqrt(self.batch_size)),)  # 25x25 patch size, add a parameter later
                        self.fabric.log_dict({"patch_size": patch_size}, step=self.step)
                        loss += self.loss_fn(y_pred, y)
                        loss += self.s3im_loss(y_pred, y, patch_size=patch_size)
                        
                    else:
                        loss = self.loss_fn(y_pred, y)

                    self.fabric.log_dict(
                        {
                            "loss": loss,
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
                        self.fabric.log_dict(
                            {"skip_alpha_value": self.model.skip_alpha.item()},
                            step=self.step,
                        )
                    self.fabric.backward(loss)
                    if self.config.clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
                    self.optim.step()
                    self.step += 1
                    if torch.isfinite(loss):
                            self._epoch_loss_sum += float(loss.item())
                            self._epoch_loss_count += 1
                self.on_angle_end()
            self.on_train_epoch_end()
        self.evaluate()
        self.save_model(last=True)
