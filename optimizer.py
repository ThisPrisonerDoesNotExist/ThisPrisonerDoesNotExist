import diffusers
import torch.optim
from diffusers.optimization import get_cosine_schedule_with_warmup

from trainingConfig import TrainingConfig


class Optimizer:
    """
    Optimizer class for the model.

    Attributes:
    - optimizer: the optimizer.
    - lr_scheduler: the learning rate scheduler.

    Methods:
    - __init__: initializes the optimizer.
    """

    def __init__(
        self,
        model: diffusers.models,
        optimizer: torch.optim,
        dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        self.optimizer = optimizer
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=TrainingConfig.lr_warmup_steps,
            num_training_steps=(len(dataloader) * TrainingConfig.num_epochs),
        )
