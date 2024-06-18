import torch
import torchvision
import torchvision.transforms.functional
from torch.utils.data import Dataset
from torchvision import transforms


class PrisonersDataset(Dataset):
    """
    Dataset class for the prisoners dataset.

    Attributes:
    - img_paths: the paths to the images.
    - transform: the transformation to apply to the images.

    Methods:
    - __len__: returns the number of images in the dataset.
    - __getitem__: returns an image at the given index.
    """

    def __init__(self, img_paths: list) -> None:
        """
        Initializes the dataset with the given image paths.

        Args:
        - img_paths (List[str]): the paths to the images.

        Returns:
        - None
        """
        self.img_paths = img_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Args:
        - None.

        Returns:
        - int: the number of images in the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns an image at the given index.

        Args:
        - idx (int): the index of the image.

        Returns:
        - Tensor: the image at the given index.
        """
        img_path = self.img_paths[idx]
        image = torchvision.io.read_image(img_path)
        image = torchvision.transforms.functional.to_pil_image(image)
        image = self.transform(image)
        return image
