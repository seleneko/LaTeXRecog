""" PREPARE.PY
"""

import os
import re
from pathlib import Path

from tqdm import tqdm

from . import path, utils

DOWNLOAD_FROM = "http://lstm.seas.harvard.edu/latex/data/"
FILES = [
    "im2latex_formulas.norm.lst",
    "im2latex_validate_filter.lst",
    "im2latex_train_filter.lst",
    "im2latex_test_filter.lst",
    "formula_images.tar.gz",
]

IMG_NUM = 100000
REPLACE_RULE = {
    "\\left(": "(",
    "\\right)": ")",
    "\\left[": "[",
    "\\right]": "]",
    "\\left{": "{",
    "\\right}": "}",
    "\\vspace { }": "",
    "\\hspace { }": "",
}


@utils.log("Downloading files")
def download_files():
    """ Download files from `DOWNLOAD_FROM`.
    """
    for filename in FILES:
        if not Path(filename).is_file():
            utils.download(DOWNLOAD_FROM + filename, filename)
        else:
            print(f"  {filename} ... Already downloaded.")


@utils.log("Extracting images")
def extract_images():
    """ Extract images from "formula_images.tar.gz" to `RAW_IMG_DIR`.
    """
    path.RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
    if len(os.listdir(path.RAW_IMG_DIR)) < IMG_NUM:
        utils.extract(path.FORMULA_TAR_GZ)
    else:
        print("  Nothing to do.")


@utils.log("Cropping images")
def crop_images():
    """ Extract regions of interest from images.
    """
    path.TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    if len(os.listdir(path.TRAIN_IMG_DIR)) < IMG_NUM:
        raw_images = list(path.RAW_IMG_DIR.glob("*.png"))
        for image in tqdm(iterable=raw_images, total=len(raw_images)):
            utils.crop_save(image, path.TRAIN_IMG_DIR / image.name)
    else:
        print("  Nothing to do.")


@utils.log("Cleaning formulas")
def clean_formulas():
    """ Do some cleaning.
    """
    rep = dict((re.escape(k), v) for k, v in REPLACE_RULE.items())
    pattern = re.compile("|".join(rep.keys()))
    if not Path(path.TRAIN_FORMULAS).is_file():
        with open(path.RAW_FORMULAS, "r") as r, open(path.TRAIN_FORMULAS, "w") as w:
            for line in r:
                w.write(pattern.sub(lambda m: rep[re.escape(m.group(0))], line))
    else:
        print("  Nothing to do.")


@utils.log("Building vocab")
def build_vocab():
    """ Build vocabulary.
    """
    if not path.VOCAB.is_file():
        all_formulas = utils.get_all_formulas(path.TRAIN_FORMULAS)
        _, train_formulas = utils.spilt(all_formulas, path.TRAIN_FILTER)
        tokenizer = utils.Vocabulary()
        tokenizer.add_formulas(train_formulas)
        utils.save_vocab(tokenizer, path.VOCAB)
    else:
        print("  Nothing to do.")


def run():
    """ Run.
    """

    path.DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(path.DATA_DIR)

    download_files()
    extract_images()
    crop_images()
    clean_formulas()
    build_vocab()


if __name__ == "__main__":
    run()
