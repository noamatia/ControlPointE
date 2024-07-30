import wandb
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from point_e.util.plotting import plot_point_cloud
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from control_shapenet import PROMPTS, SOURCE_LATENTS, TARGET_LATENTS
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config

TEXTS = "texts"
SOURCE = "source"
TARGET = "target"
UPSAMPLE = "upsample"
MODEL_NAME = "base40M-textvec"
LATENT_TYPES = [SOURCE, TARGET]


class ControlPointE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        timesteps: int,
        num_points: int,
        batch_size: int,
        dev: torch.device,
        switch_prob: float,
        cond_drop_prob: float,
        val_data_loader: DataLoader,
    ):
        super().__init__()
        self.lr = lr
        self.dev = dev
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.switch_prob = switch_prob
        self._init_model(cond_drop_prob, num_points)
        # self._init_val_data(val_data_loader)

    def _init_model(self, cond_drop_prob, num_points):
        self.diffusion = diffusion_from_config(
            {**DIFFUSION_CONFIGS[MODEL_NAME], "timesteps": self.timesteps}
        )
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[UPSAMPLE])
        self.model = model_from_config(
            {**MODEL_CONFIGS[MODEL_NAME], "cond_drop_prob": cond_drop_prob}, self.dev
        )
        self.model.load_state_dict(load_checkpoint(MODEL_NAME, self.dev))
        self.model.create_control_layers()
        upsampler_model = model_from_config(MODEL_CONFIGS[UPSAMPLE], self.dev)
        upsampler_model.eval()
        upsampler_model.load_state_dict(load_checkpoint(UPSAMPLE, self.dev))
        self.sampler = PointCloudSampler(
            device=self.dev,
            guidance_scale=[3.0, 0.0],
            aux_channels=["R", "G", "B"],
            model_kwargs_key_filter=(TEXTS, ""),
            models=[self.model, upsampler_model],
            num_points=[num_points, 4096 - num_points],
            diffusions=[self.diffusion, upsampler_diffusion],
        )

    def _init_val_data(self, val_data_loader):
        log_data = {lt: [] for lt in LATENT_TYPES}
        for batch in val_data_loader:
            for prompt, source_latent, target_latent in zip(
                batch[PROMPTS], batch[SOURCE_LATENTS], batch[TARGET_LATENTS]
            ):
                latents = [source_latent, target_latent]
                for name, latent in zip(LATENT_TYPES, latents):
                    log_data[name].append(self._plot([latent], prompt))
        wandb.log(log_data, step=None)

    def _plot(self, samples, prompt):
        pc = self.sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, theta=np.pi * 3 / 2)
        img = wandb.Image(fig, caption=prompt)
        plt.close()
        return img

    def _sample_t(self):
        return (
            torch.tensor(random.sample(range(self.timesteps), self.batch_size))
            .to(self.dev)
            .detach()
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        prompts, source_latents, target_latents = (
            batch[PROMPTS],
            batch[SOURCE_LATENTS],
            batch[TARGET_LATENTS],
        )
        if self.switch_prob is not None and random.random() < self.switch_prob:
            source_latents = target_latents.clone()
        terms = self.diffusion.training_losses(
            model=self.model,
            t=self._sample_t(),
            x_start=target_latents,
            model_kwargs={TEXTS: prompts, "guidance": source_latents},
        )
        loss = terms["loss"].mean()
        wandb.log({"loss": loss.item()}, step=None)
        return loss

    def validation_step(self, batch, batch_idx):
        assert batch_idx == 0
        outputs = []
        with torch.no_grad():
            for prompt, source_latent in zip(batch[PROMPTS], batch[SOURCE_LATENTS]):
                samples = self._sample(source_latent, [prompt], 1)
                outputs.append(self._plot(samples, prompt))
        wandb.log({"output": outputs}, step=None)

    def _sample(self, source_latents, prompts, batch_size):
        samples = None
        for x in tqdm(self.sampler.sample_batch_progressive(
            batch_size=batch_size,
            model_kwargs={TEXTS: prompts},
            guidances=[source_latents, None],
        )):
            samples = x
        return samples
