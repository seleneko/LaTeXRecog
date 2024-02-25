""" METRIC.PY
"""

from typing import Tuple

import editdistance
import torch
from torch import Tensor
from torchmetrics import Metric

from ..utils import path, utils


class EditDistance(Metric):
    """ Edit Distance.
    """

    def __init__(self) -> None:
        super().__init__()
        tokenizer = utils.load_vocab(path.VOCAB)
        self.ignore_indices = {
            tokenizer.sos,
            tokenizer.eos,
            tokenizer.pad,
        }
        self.error = torch.tensor(0.0)
        self.total = torch.tensor(0)
        self.match = torch.tensor(0)

    def update(self, *args, **_kwargs) -> None:
        self._update(*args)

    def _update(self, preds: Tensor, tgts: Tensor) -> None:
        num = preds.shape[0]
        for i in range(num):
            pred = [t for t in preds[i].tolist() if t not in self.ignore_indices]
            target = [t for t in tgts[i].tolist() if t not in self.ignore_indices]
            distance = editdistance.distance(pred, target)
            self.error += distance / max(len(pred), len(target), 1)
            self.match += distance == 0
        self.total += num

    def compute(self) -> Tuple[Tensor, Tensor]:
        return 1 - self.error / self.total, self.match / self.total
