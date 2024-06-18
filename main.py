from torch.utils.data import DataLoader

from pathsLoader import load_paths_prisoners_dataset
from prisonersDataset import PrisonersDataset
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

# Create noise scheduler

# Train and save model
