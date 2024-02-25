""" UTILS.PY
"""

import pickle
import tarfile
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

Formula = List[str]


def log(doing: str):
    """ Log decorator.

    Args:
        doing (str): What is doing now.
    """

    def decorator(inner):
        def wrapper(*args, **kw):
            print(f"{doing}...")
            inner(*args, **kw)
            print("Done.")

        return wrapper

    return decorator


class Progress(tqdm):
    """ Progress Bar when Downloading.
    """

    def update_to(self, blocks=1, bsize=1, tsize=None) -> None:
        """ Inform the progress bar how many data have been downloaded.

        Args:
            blocks (int, optional): number of blocks. defaults to 1.
            bsize (int, optional): block size. defaults to 1.
            tsize (int, optional): total size. defaults to None.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def download(url: Path, filename: Path) -> None:
    """ Download a file from url to filename.

    Args:
        url (Path): the url that will be retrieved.
        filename (Path): the name of the received file.
    """
    with Progress(unit="B", unit_scale=True) as f:
        f.set_description(filename)
        urlretrieve(url, filename, reporthook=f.update_to)


def extract(filename: Path) -> None:
    """ Extract a .gz file.

    Args:
        filename (Path): the input file.
    """
    with tarfile.open(filename, "r") as g:
        for image in tqdm(iterable=g.getmembers(), total=len(g.getmembers())):
            g.extract(image)


def crop_save(filename: Path, save_to: Path, padding: int = 8):
    """ Crop the input image and save it.

    Args:
        filename (Path): the input image.
        save_to (Path): save to.
        padding (int, optional): padding. defaults to 8.
    """
    with open(filename, "rb") as f:
        image = Image.open(f).convert("RGBA")

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("L")

    arr = 255 - np.array(new_image)

    row_nonzero = np.argwhere(np.sum(arr, axis=1) != 0)[:, 0]
    col_nonzero = np.argwhere(np.sum(arr, axis=0) != 0)[:, 0]
    if row_nonzero.size == 0 or col_nonzero.size == 0:
        return

    y_start, y_end = row_nonzero[0] - padding, row_nonzero[-1] + padding
    x_start, x_end = col_nonzero[0] - padding, col_nonzero[-1] + padding
    cropped = arr[y_start:y_end, x_start:x_end]

    cropped_image = Image.fromarray(255 - cropped).convert("L")
    cropped_image.save(save_to)


def get_all_formulas(filename: Path) -> List[Formula]:
    """ Returns all the formulas in the formula file.

    Args:
        filename (Path): file name.

    Returns:
        List[Formula]: list of all formulas.
    """

    with open(filename) as f:
        all_formulas = [formula.strip("\n").split() for formula in f.readlines()]
    return all_formulas


def spilt(
    all_formulas: List[Formula], filter_file: Path,
) -> Tuple[List[str], List[Formula]]:
    """ Returns train formulas in the formula file.

    Args:
        all_formulas (List[Formula]): list of all formulas.
        filter_file (Path): filter file.

    Returns:
        List[Formula]: list of train formulas.
    """
    image_names = []
    formulas = []
    with open(filter_file) as f:
        for line in f:
            img_name, formula_idx = line.strip("\n").split()
            image_names.append(img_name)
            formulas.append(all_formulas[int(formula_idx)])
    return image_names, formulas


class Vocabulary:
    """ Vocabulary.
    """

    def __init__(self) -> None:
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.idx = 0
        self.pad = self.add_token("<pad>")
        self.sos = self.add_token("<sos>")
        self.eos = self.add_token("<eos>")
        self.unk = self.add_token("<unk>")

    def add_token(self, token: str) -> int:
        """ Add one token to the vocabulary.

        Args:
            token (str): the token to be added.

        Returns:
            int: the index of the input token.
        """
        if not token in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1
        return self.token2idx[token]

    def add_formulas(self, formulas: List[Formula], threshold: int = 4) -> None:
        """ Create a mapping from tokens to indices and vice versa.

        Args:
            formulas (List[Formula]): list of formulas.
            threshold (int, optional): tokens that appear fewer than it will not be
                included in the mapping. defaults to 4.
        """
        counter = Counter()
        for formula in formulas:
            counter.update(formula)
        tokens = [token for token, cnt in counter.items() if cnt >= threshold]
        for token in tokens:
            self.add_token(token)

    def formula_to_indices(self, formula: Formula) -> List[int]:
        """ Formula to indices.

        Args:
            formula (Formula): the input formula.

        Returns:
            List[int]: the indices.
        """
        return (
            [self.sos]
            + [self.token2idx.get(token, self.unk) for token in formula]
            + [self.eos]
        )

    def indices_to_formula(self, indices: List[int]) -> Formula:
        """ Indices to formula.

        Args:
            indices (list): the input indices.

        Returns:
            list: the formula.
        """
        return [
            self.idx2token.get(idx, "<unk>")
            for idx in indices
            if not idx in [self.pad, self.sos, self.eos, self.unk]
        ]

    def __call__(self, token):
        return self.token2idx[token if token in self.token2idx else "<unk>"]

    def __len__(self):
        return len(self.token2idx)


def save_vocab(vocab: Vocabulary, filename: Path):
    """ Save vocabulary.

    Args:
        vocab (Vocabulary): vocabulary.
        filename (Path): path of the vocab file.
    """
    with open(filename, "wb") as f:
        pickle.dump(vocab, f)


class RenameUnpickler(pickle.Unpickler):
    """ Rename Unpickler.
    """

    def find_class(self, module, name):
        renamed_module = module
        if module == "Vocabulary":
            renamed_module = "utils.Vocabulary"

        return super().find_class(renamed_module, name)


def renamed_load(file_obj):
    """ Renamed load.
    """
    return RenameUnpickler(file_obj).load()


def load_vocab(filename: Path) -> Vocabulary:
    """ Load vocabulary.

    Args:
        filename (Path): path of the vocab file.

    Returns:
        Vocabulary: vocabulary.
    """
    with open(filename, "rb") as f:
        vocab: Vocabulary = pickle.load(f)
    return vocab


class ImageDataset(Dataset):
    """ Image Dataset.
    """

    def __init__(
        self,
        root_dir: Path,
        image_names: List[str],
        formulas: List[List[str]],
        transform: Callable,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_names = image_names
        self.formulas = formulas
        self.transform = transform

    def __len__(self) -> int:
        return len(self.formulas)

    def __getitem__(self, index) -> Tuple[Image.Image, List[List[str]]]:
        image_name, formula = self.image_names[index], self.formulas[index]
        image_path = self.root_dir / image_name
        if image_path.is_file():
            with open(image_path, "rb") as f:
                image = Image.open(f).convert("L")
        else:
            image = Image.fromarray(np.full((64, 128), 255, dtype=np.uint8))
            formula = []
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
        return image, formula
