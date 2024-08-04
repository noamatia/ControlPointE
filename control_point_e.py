import wandb
import torch
import random
import numpy as np
from lora import LoRA
from tqdm import tqdm
import torch.optim as optim
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from point_e.util.plotting import plot_point_cloud
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from shapetalk import (
    PROMPTS,
    SOURCE_LATENTS,
    TARGET_LATENTS,
    SWITCH_PROMPTS,
    NEGATIVE_LATENTS,
    POSITIVE_LATENTS,
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
)

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
        val_data_loader: DataLoader = None,
    ):
        super().__init__()
        self.lr = lr
        self.dev = dev
        self.theta = np.pi * 3 / 2
        self.batch_size = batch_size
        self.switch_prob = switch_prob
        self._init_model(num_points)
        self._init_val_data(val_data_loader)

    def build_lora(self, val_data_loader):
        self.model.freeze_all_parameters()
        self.lora = LoRA(self.model).to(self.dev)
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
        self.lora = None

    def _init_val_data(self, val_data_loader):
        if val_data_loader is None:
            return
        log_data = {lt: [] for lt in LATENT_TYPES}
        for batch in val_data_loader:
            if self.lora is None:
                for prompt, source_latent, target_latent in tqdm(
                    zip(batch[PROMPTS], batch[SOURCE_LATENTS], batch[TARGET_LATENTS]),
                    total=len(batch[PROMPTS]),
                    desc="Initating val data",
                ):
                    latents = [source_latent, target_latent]
                    for latent_type, latent in zip(LATENT_TYPES, latents):
                        samples = self.sampler.sample_batch(
                            batch_size=1,
                            model_kwargs={},
                            prev_samples=latent.unsqueeze(0),
                        )
                        log_data[latent_type].append(self._plot(samples, prompt))
            else:
                for (
                    negative_prompt,
                    positive_prompt,
                    negative_latent,
                    positive_latent,
                ) in tqdm(
                    zip(
                        batch[NEGATIVE_PROMPTS],
                        batch[POSITIVE_PROMPTS],
                        batch[NEGATIVE_LATENTS],
                        batch[POSITIVE_LATENTS],
                    ),
                    total=len(batch[NEGATIVE_PROMPTS]),
                    desc="Initating val data",
                ):
                    prompts = [negative_prompt, positive_prompt]
                    latents = [negative_latent, positive_latent]
                    for i, latent in enumerate(latents):
                        samples = self._sample(prev_samples=latent.unsqueeze(0))
                        for j, latent_type in enumerate(LATENT_TYPES):
                            log_data[latent_type].append(
                                self._plot(samples, prompts[int(i == j)])
                            )
        wandb.log(log_data, step=None)

    def _plot(self, samples, prompt):
        pc = self.sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, theta=self.theta, color_by_distance=True)
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
        return optim.Adam(
            (
                self.parameters()
                if self.lora is None
                else self.lora.prepare_optimizer_params()
            ),
            lr=self.lr,
        )

    def training_step(self, batch, batch_idx):
        if self.lora is None:
            prompts, source_latents, target_latents, switch_prompts = (
                batch[PROMPTS],
                batch[SOURCE_LATENTS],
                batch[TARGET_LATENTS],
                batch[SWITCH_PROMPTS],
            )
            if self.switch_prob and random.random() < self.switch_prob:
                source_latents = target_latents
                prompts = switch_prompts
            terms = self.diffusion.training_losses(
                model=self.model,
                t=self._sample_t(),
                x_start=target_latents,
                model_kwargs={TEXTS: prompts, "guidance": source_latents},
            )
            loss = terms["loss"].mean()
            log_data = {"loss": loss.item()}
        else:
            negative_prompts, positive_prompts, negative_latents, positive_latents = (
                batch[NEGATIVE_PROMPTS],
                batch[POSITIVE_PROMPTS],
                batch[NEGATIVE_LATENTS],
                batch[POSITIVE_LATENTS],
            )
            self.lora.set_lora_slider(-1)
            with self.lora:
                terms = self.diffusion.training_losses(
                    model=self.model,
                    t=self._sample_t(),
                    x_start=positive_latents,
                    model_kwargs={
                        TEXTS: negative_prompts,
                        "guidance": positive_latents,
                    },
                )
            negative_loss = terms["loss"].mean()
            self.lora.set_lora_slider(1)
            with self.lora:
                terms = self.diffusion.training_losses(
                    model=self.model,
                    t=self._sample_t(),
                    x_start=negative_latents,
                    model_kwargs={
                        TEXTS: positive_prompts,
                        "guidance": negative_latents,
                    },
                )
            positive_loss = terms["loss"].mean()
            loss = (negative_loss + positive_loss) / 2
            log_data = {
                "loss": loss.item(),
                "negative_loss": negative_loss.item(),
                "positive_loss": positive_loss.item(),
            }
        wandb.log(log_data, step=None)
        return loss

    def _sample(self, prompt=None, guidance=None, prev_samples=None):
        return self.sampler.sample_batch(
            batch_size=1,
            prev_samples=prev_samples,
            model_kwargs={TEXTS: [prompt]} if prompt is not None else {},
            guidances=[guidance, None] if guidance is not None else None,
        )

    def validation_step(self, batch, batch_idx):
        assert batch_idx == 0
        outputs = []
        with torch.no_grad():
            if self.lora is None:
                for prompt, source_latent in zip(batch[PROMPTS], batch[SOURCE_LATENTS]):
                    samples = self._sample(prompt, source_latent)
                    outputs.append(self._plot(samples, prompt))
            else:
                for (
                    negative_prompt,
                    positive_prompt,
                    negative_latent,
                    positive_latent,
                ) in zip(
                    batch[NEGATIVE_PROMPTS],
                    batch[POSITIVE_PROMPTS],
                    batch[NEGATIVE_LATENTS],
                    batch[POSITIVE_LATENTS],
                ):
                    self.lora.set_lora_slider(-1)
                    with self.lora:
                        samples = self._sample(negative_prompt, positive_latent)
                    outputs.append(self._plot(samples, negative_prompt))
                    self.lora.set_lora_slider(1)
                    with self.lora:
                        samples = self._sample(positive_prompt, negative_latent)
                    outputs.append(self._plot(samples, positive_prompt))
        wandb.log({"output": outputs}, step=None)
