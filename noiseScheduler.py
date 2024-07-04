import torch
from diffusers import DDPMScheduler

from prisonersDataset import PrisonersDataset


class Scheduler:
    """
    Scheduler class for the noise.

    Attributes:
    - noise_scheduler: the noise scheduler.
    - noise: the noise tensor.
    - timesteps: the timesteps.

    Methods:
    - __init__: initializes the scheduler.

    """

    def __init__(self, sample_image: torch.Tensor, lr_warmup_steps: int) -> None:

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.noise = torch.randn(sample_image.shape)
        self.timesteps = torch.LongTensor([50])
