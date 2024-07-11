import torch
from torch.utils.data import DataLoader

from thisprisonerdoesnotexist.src.fitModel import FitModel
from thisprisonerdoesnotexist.src.noiseScheduler import Scheduler
from thisprisonerdoesnotexist.src.optimizer import Optimizer
from thisprisonerdoesnotexist.src.pathsLoader import load_paths
from thisprisonerdoesnotexist.src.prisonersDataset import PrisonersDataset
from thisprisonerdoesnotexist.src.prisonersModel import PrisonersModel
from thisprisonerdoesnotexist.src.trainingConfig import TrainingConfig

# Load paths of files
paths = load_paths(TrainingConfig.training_data_dir)
# Create dataset
dataset = PrisonersDataset(paths)
# Create data loader
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
# Create model
model = PrisonersModel(
    sample_size=TrainingConfig.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
).model
# Create noise scheduler
scheduler = Scheduler(sample_image=dataset[0].unsqueeze(0), lr_warmup_steps=1000)

# Optimizer
optimizer = Optimizer(
    model=model,
    dataloader=dataloader,
    optimizer=torch.optim.AdamW(model.parameters(), lr=TrainingConfig.learning_rate),
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