""" DATALOADER.PY
"""

from typing import Dict, List, Tuple

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import path, prepare, utils

# apply affine, Gaussian noise and Gaussian filter to training images.
TRANSFORM = {
    "train": A.Compose(
        [
            A.Affine(scale=(0.8, 1.0), cval=255),
            A.GaussNoise(var_limit=(10, 100)),
            A.GaussianBlur(blur_limit=(1, 1)),
            ToTensorV2(),
        ]
    ),
    "val/test": ToTensorV2(),
}

tokenizer: utils.Vocabulary = utils.load_vocab(path.VOCAB)


def collate_fn(
    batch: List[Tuple[Tensor, utils.Formula]]
) -> Tuple[List[Tensor], List[List[int]]]:
    """ Collate function.

    Args:
        batch (List[Tuple[Tensor, utils.Formula]]): a batch of images and formulas.

    Returns:
        Tuple[List[Tensor], List[List[int]]]: a tuple of batched images and batches formulas.
    """
    images, formulas = zip(*batch)
    batch_size = len(images)
    max_length = max(len(formula) for formula in formulas)
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)
    gray_scale_images = torch.zeros((batch_size, 1, max_height, max_width))
    batched_indices = torch.zeros((batch_size, max_length + 2), dtype=torch.long)
    for i in range(batch_size):
        gray_scale_images[i, :, : images[i].shape[1], : images[i].shape[2]] = images[i]
        indices = tokenizer.formula_to_indices(formulas[i])
        batched_indices[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
    return gray_scale_images, batched_indices


class LitDataLoader(LightningDataModule):
    """ Lightning Data Loader for Im2LaTeX Dataset.
    """

    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool) -> None:
        super().__init__()
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn,
            "drop_last": True,
        }
        self.dataset: Dict[utils.ImageDataset] = {}

    def prepare_data(self):
        prepare.run()
        all_formulas = utils.get_all_formulas(path.TRAIN_FORMULAS)
        train_image_names, train_formulas = utils.spilt(all_formulas, path.TRAIN_FILTER)
        self.dataset["train"] = utils.ImageDataset(
            path.TRAIN_IMG_DIR, train_image_names, train_formulas, TRANSFORM["train"],
        )
        val_image_names, val_formulas = utils.spilt(all_formulas, path.VAL_FILTER)
        self.dataset["val"] = utils.ImageDataset(
            path.TRAIN_IMG_DIR, val_image_names, val_formulas, TRANSFORM["val/test"],
        )
        test_image_names, test_formulas = utils.spilt(all_formulas, path.TEST_FILTER)
        self.dataset["test"] = utils.ImageDataset(
            path.TRAIN_IMG_DIR, test_image_names, test_formulas, TRANSFORM["val/test"],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], shuffle=True, **self.loader_config)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["val"], shuffle=False, **self.loader_config)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["test"], shuffle=False, **self.loader_config)

    def predict_dataloader(self) -> DataLoader:
        return None
