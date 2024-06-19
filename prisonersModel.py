import diffusers.models
from diffusers import DDPMPipeline, UNet2DModel
from diffusers.utils import make_image_grid


class PrisonersModel:
    """
    This class creates model to training

    """

    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels: tuple,
    ) -> None:
        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
