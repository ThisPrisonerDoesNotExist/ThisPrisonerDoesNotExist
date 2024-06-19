import torch
from torch.utils.data import DataLoader

from fitModel import FitModel
from noiseScheduler import Scheduler
from optimizer import Optimizer
from pathsLoader import load_paths_prisoners_dataset
from prisonersDataset import PrisonersDataset
from prisonersModel import PrisonersModel
from trainingConfig import TrainingConfig
from unzip import unzip_prisoners_dataset

# Download dataset with: git clone https://huggingface.co/datasets/MGKK/Prisonersi
# Unzip dataset
unzip_prisoners_dataset()
# Load paths of files
paths = load_paths_prisoners_dataset()
# Load training config
trainingConfig = TrainingConfig()
# Create dataset
dataset = PrisonersDataset(paths)
# Create data loader
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
# Create model
model = PrisonersModel(
    sample_size=trainingConfig.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
).model
# Create noise scheduler
scheduler = Scheduler(sample_image=dataset[0].unsqueeze(0), lr_warmup_steps=1000)

# Optimazer
optimizer = Optimizer(
    model=model,
    dataloader=dataloader,
    optimizer=torch.optim.AdamW(model.parameters(), lr=trainingConfig.learning_rate),
)
# Train and save model

fit = FitModel(
    model=model,
    scheduler=scheduler.noise_scheduler,
    optimizer=optimizer.optimizer,
    dataloader=dataloader,
    lr_scheduler=optimizer.lr_scheduler,
)
fit.train_loop()
