import diffusers.models
from diffusers import DDPMPipeline, UNet2DModel
from diffusers.utils import make_image_grid


class PrisonersModel:
    """
    This class creates model for training

    Attributes:
    - model: the model.

    Methods:
    - __init__: initializes the model.
    """

    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels: tuple,
    ) -> None:
        """
        Initializes the model.

        Args:
        - sample_size (int): the sample size.
        - in_channels (int): the input channels.
        - out_channels (int): the output channels.
        - layers_per_block (int): the layers per block.
        - block_out_channels (tuple): the block out channels.
        """
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
