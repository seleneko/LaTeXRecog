""" PATH.PY
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "dataset"
RAW_IMG_DIR = DATA_DIR / "formula_images"
TRAIN_IMG_DIR = DATA_DIR / "train_images"
FORMULA_TAR_GZ = DATA_DIR / "formula_images.tar.gz"
RAW_FORMULAS = DATA_DIR / "im2latex_formulas.norm.lst"
TRAIN_FORMULAS = DATA_DIR / "im2latex_formulas.norm.new.lst"
TRAIN_FILTER = DATA_DIR / "im2latex_train_filter.lst"
VAL_FILTER = DATA_DIR / "im2latex_validate_filter.lst"
TEST_FILTER = DATA_DIR / "im2latex_test_filter.lst"
VOCAB = DATA_DIR / "vocab"

SRC_DIR = PROJECT_DIR / "src"
