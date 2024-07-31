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
        num_points: int,
        batch_size: int,
        dev: torch.device,
        switch_prob: float,
        empty_prompt: bool,
        val_data_loader: DataLoader,
    ):
        super().__init__()
        self.lr = lr
        self.dev = dev
        self.theta = np.pi * 3 / 2
        self.batch_size = batch_size
        self.switch_prob = switch_prob
        self.empty_prompt = empty_prompt
        self._init_model(num_points)
        self._init_val_data(val_data_loader)

    def _init_model(self, num_points):
        self.diffusion = diffusion_from_config(DIFFUSION_CONFIGS[MODEL_NAME])
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[UPSAMPLE])
        self.model = model_from_config(MODEL_CONFIGS[MODEL_NAME], self.dev)
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
        for batch in tqdm(
            val_data_loader, total=len(val_data_loader), desc="Initating val data"
        ):
            for prompt, source_latent, target_latent in zip(
                batch[PROMPTS], batch[SOURCE_LATENTS], batch[TARGET_LATENTS]
            ):
                latents = [source_latent, target_latent]
                for name, latent in zip(LATENT_TYPES, latents):
                    samples = self.sampler.sample_batch(
                        batch_size=1,
                        model_kwargs={},
                        prev_samples=latent.unsqueeze(0),
                    )
                    log_data[name].append(self._plot(samples, prompt))
        wandb.log(log_data, step=None)

    def _plot(self, samples, prompt):
        pc = self.sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, theta=self.theta)
        img = wandb.Image(fig, caption=prompt)
        plt.close()
        return img

    def _sample_t(self):
        return (
            torch.tensor(
                random.sample(range(len(self.diffusion.betas)), self.batch_size)
            )
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
            source_latents = target_latents
            if self.empty_prompt:
                prompts = ["" for _ in prompts]
        terms = self.diffusion.training_losses(
            model=self.model,
            t=self._sample_t(),
            x_start=target_latents,
            model_kwargs={TEXTS: prompts, "guidance": source_latents},
        )
        loss = terms["loss"].mean()
        wandb.log(
            {"loss": loss.item()},
            step=None,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        assert batch_idx == 0
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for prompt, source_latent in zip(batch[PROMPTS], batch[SOURCE_LATENTS]):
                samples = self.sampler.sample_batch(
                    batch_size=1,
                    model_kwargs={TEXTS: [prompt]},
                    guidances=[source_latent, None],
                )
                outputs.append(self._plot(samples, prompt))
        wandb.log({"output": outputs}, step=None)
        self.model.train()
