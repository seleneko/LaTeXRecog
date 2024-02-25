""" PE.PY
"""


import torch
import torch.nn as nn
from torch import Tensor


def make_pe2d(d_model: int, max_h: int, max_w: int) -> Tensor:
    """ Make 2D Positional Encoding.

    Args:
        d_model (int): model dim.
        max_h (int): max height.
        max_w (int): max width.

    Returns:
        Tensor: positional encoding.
    """
    pe_h = make_pe1d(d_model // 2, max_h)  # [max_h, 1, d_model // 2]
    pe_h = pe_h.permute(2, 0, 1).expand(-1, -1, max_w)  # [d_model // 2, max_h, max_w]
    pe_w = make_pe1d(d_model // 2, max_w)  # [max_w, 1, d_model // 2]
    pe_w = pe_w.permute(2, 1, 0).expand(-1, max_h, -1)  # [d_model // 2, max_h, max_w]
    pe = torch.cat([pe_h, pe_w], dim=0)  # [d_model, max_h, max_w]
    return pe


def make_pe1d(d_model: int, max_len: int) -> Tensor:
    """ Make 1D Positional Encoding.

    Args:
        d_model (int): model dim.
        max_len (int): max length.

    Returns:
        Tensor: positional encoding.
    """
    pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
    return pe


class PositionalEncoding2d(nn.Module):
    """ 2D Positional Encoding.
    """

    def __init__(self, d_model: int, max_h: int = 2048, max_w: int = 2048) -> None:
        super().__init__()
        self.pe = make_pe2d(d_model, max_h, max_w)

    def forward(self, x: Tensor) -> Tensor:
        """ Forward.
        """
        return x + self.pe[:, : x.shape[2], : x.shape[3]].type_as(x)


class PositionalEncoding1d(nn.Module):
    """ 1D Positional Encoding.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.pe = make_pe1d(d_model, max_len)

    def forward(self, x: Tensor) -> Tensor:
        """ Forward.
        """
        return self.dropout(x + self.pe[: x.shape[0]].type_as(x))
