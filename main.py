""" MAIN.PY
"""

from pytorch_lightning.utilities.cli import LightningCLI

from src.data.dataloader import LitDataLoader
from src.model.litmodel import LitModel

if __name__ == "__main__":
    LightningCLI(LitModel, LitDataLoader)
