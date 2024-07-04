import os
from pathlib import Path

import diffusers
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm.auto import tqdm

from trainingConfig import TrainingConfig


class FitModel:
    def __init__(
        self,
        model: diffusers.models,
        scheduler: diffusers.schedulers.DDPMScheduler,
        optimizer: torch.optim,
        dataloader: torch.utils.data.dataloader.DataLoader,
        lr_scheduler: diffusers.optimization,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.lr_scheduler = lr_scheduler
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")

    def evaluate(
        self,
        config: type[TrainingConfig],
        epoch: int,
        pipeline: diffusers.DDPMPipeline,
        columns: int = 1,
        rows: int = 1,
    ) -> None:
        """Evaluation model for each epoch."""
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.Generator(device="cuda").manual_seed(config.seed),
            # Use a separate torch generator to avoid rewinding the random state of the main training loop
        ).images
        image_grid = make_image_grid(images, rows=rows, cols=columns)

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def train_loop(self) -> None:
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=TrainingConfig.mixed_precision,
            gradient_accumulation_steps=TrainingConfig.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(TrainingConfig.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if TrainingConfig.output_dir is not None:
                os.makedirs(TrainingConfig.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.lr_scheduler
        )

        global_step = 0

        # Now you train the model
        for epoch in range(TrainingConfig.num_epochs):
            progress_bar = tqdm(
                total=len(train_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (bs,),
                    device=clean_images.device,
                    dtype=torch.int64,
                )

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    ssim = self.ssim_metric(noise_pred, noise).item()

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    "ssim": ssim,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model), scheduler=self.scheduler
                )

                if (
                    (epoch + 1) % TrainingConfig.save_image_epochs == 0
                    or epoch == TrainingConfig.num_epochs - 1
                ):
                    self.evaluate(TrainingConfig, epoch, pipeline)
                    pipeline.save_pretrained(TrainingConfig.output_dir)
                    accelerator.save_model(model, TrainingConfig.output_dir)
