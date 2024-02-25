""" MODEL.PY
"""

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor

from .pe import PositionalEncoding1d, PositionalEncoding2d


@dataclass
class LSTMConfig:
    """ LSTM Config.
    """

    hidden_size: int
    num_layers: int


class ResNetRNN(nn.Module):
    """ ResNet-LSTM Model.
    """

    def __init__(
        self,
        embed_size: int,
        num_classes: int,
        lstm_config: LSTMConfig,
        max_output_len: int,
    ):
        super().__init__()
        self.max_output_len = max_output_len

        # encoder:
        resnet = torchvision.models.resnet18()
        resnet.fc = nn.Sequential(nn.Linear(512, embed_size))
        self.resnet = resnet
        self.bn = nn.BatchNorm1d(embed_size)

        # decoder:
        self.embedding = nn.Embedding(num_classes, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_config.hidden_size, num_classes)

    def forward(self, image: Tensor, tgt: Tensor) -> Tensor:
        """ Forward.

        Notations:
            N: batch size.
            C: num of channels.
            H: height of the image.
            W: width of the image.

        Args:
            image (Tensor): input image in gray scale, shape = [N, C=1, H, W].
            tgt (Tensor): target formula indices.

        Returns:
            Tensor: forward prop result.
        """
        feature = self.encode(image)  # [N, src_size, d_model]
        output = self.decode(feature, tgt)  # [N, tgt_size, num_classes]
        return output

    def encode(self, src: Tensor) -> Tensor:
        """ Encoder.

        Args:
            src (Tensor): input image in gray scale, shape = [N, C=1, H, W].

        Returns:
            Tensor: image feature.
        """
        if src.shape[1] == 1:  # gray scale => RGB
            src = src.repeat(1, 3, 1, 1)  # [N, C=3, H, W]
        src = self.resnet(src)  # [N, embed_size]
        src = self.bn(src)  # [N, embed_size]
        return src

    def decode(self, feature: Tensor, tgt: Tensor):
        """ Decoder.

        Args:
            feature (Tensor): images features, shape = [N, embed_size].
            tgt (Tensor): target, shape = [N, tgt_size].

        Return:
            Tensor: forward prop result, shape = [N, tgt_size, num_classes].
        """
        feature = feature.unsqueeze(1)  # [N, 1, embed_size]
        embed = self.embedding(tgt)  # [N, tgt_size, embed_size]
        embed = torch.cat((feature, embed), 1)  # [N, n + 1, embed_size]
        hidden, _ = self.lstm(embed)  # [N, tgt_size, hidden_size]
        output = self.fc(hidden[0])  # [N, tgt_size, num_classes]
        return output

    def sample(self, image: Tensor):
        """ Generate Formula.

        Args:
            image (Tensor): input image in gray scale, shape = [N, C=1, H, W].

        Returns:
            Tensor: predicted formula.
        """
        output = []
        states = None
        inputs = image.unsqueeze(1)  # [batch_size, 1, embed_size]
        for _ in range(self.max_output_len):
            hiddens, states = self.lstm(inputs, states)  # h: [batch_size, hidden_size]
            outputs = self.fc(hiddens.squeeze(1))  # [batch_size, vocab_size]
            _, predicted = outputs.max(1)  # [batch_size]
            output.append(predicted)
            inputs = self.embedding(predicted)  # [batch_size, embed_size]
            inputs = inputs.unsqueeze(1)  # [batch_size, 1, embed_size]

        output = torch.stack(output, 1)  # [batch_size, max_seq_length]
        return output


@dataclass
class TransformerConfig:
    """[summary]
    """

    nhead: int
    dim_feedforward: int
    dropout: float
    num_layers: int


class ResNetTransformer(nn.Module):  # pylint: disable = too-many-instance-attributes
    """ ResNet-Transformer Model.
    """

    def __init__(  # pylint: disable = too-many-arguments
        self,
        d_model: int,
        num_classes: int,
        t_config: TransformerConfig,
        ignore_indices: Dict[str, int],
        max_output_len: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.ignore_indices = ignore_indices
        self.max_output_len = max_output_len

        # encoder:
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*(list(resnet.children())[:-3]))
        self.bottleneck = nn.Conv2d(256, self.d_model, 1)
        self.image_pe = PositionalEncoding2d(d_model)

        # decoder:
        self.embedding = nn.Embedding(num_classes, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=t_config.nhead,
                dim_feedforward=t_config.dim_feedforward,
                dropout=t_config.dropout,
                batch_first=True,
            ),
            num_layers=t_config.num_layers,
        )
        self.word_pe = PositionalEncoding1d(d_model)
        self.tgt_mask = torch.triu(torch.ones(4096, 4096) * float("-inf"), diagonal=1)
        self.fc = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """ Initialize weights.

        Ref:
            https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

        nn.init.kaiming_normal_(
            self.bottleneck.weight.data, a=0, mode="fan_out", nonlinearity="relu",
        )
        bound = 1 / math.sqrt(128)
        nn.init.normal_(self.bottleneck.bias, -bound, bound)

    def forward(self, image: Tensor, tgt: Tensor) -> Tensor:
        """ Forward.

        Notations:
            N: batch size.
            C: num of channels.
            H: height of the image.
            W: width of the image.

        Args:
            image (Tensor): input image in gray scale, shape = [N, C=1, H, W].
            tgt (Tensor): target formula indices.

        Returns:
            Tensor: forward prop result.
        """
        memory = self.encode(image)  # [N, src_size, d_model]
        output = self.decode(tgt, memory)  # [N, tgt_size, num_classes]
        output = output.permute(0, 2, 1)  # [N, num_classes, tgt_size]
        return output

    def encode(self, src: Tensor) -> Tensor:
        """ Encoder.

        Args:
            src (Tensor): input image in gray scale, shape = [N, C=1, H, W].

        Returns:
            Tensor: image feature.
        """
        if src.shape[1] == 1:  # gray scale => RGB
            src = src.repeat(1, 3, 1, 1)  # [N, C=3, H, W]
        src = self.backbone(src)  # [N, d_model, H' := H//32, W' := W//32]
        src = self.bottleneck(src)
        src = self.image_pe(src)  # [N, d_model, H', W']
        src = src.flatten(start_dim=2)  # [N, d_model, src_size := H'*W']
        src = src.permute(0, 2, 1)  # [N, src_size, d_model]
        return src

    def decode(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """ Decoder.

        Args:
            tgt (Tensor): target, shape = [N, tgt_size].
            memory (Tensor): memory, shape = [N, src_size, d_model].

        Returns:
            Tensor: forward prop result, shape = [N, tgt_size, num_classes].
        """
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)  # [N, tgt_size, d_model]
        tgt = self.word_pe(tgt)  # [N, tgt_size, d_model]
        mask_size = tgt.shape[1]
        tgt_mask = self.tgt_mask[:mask_size, :mask_size].type_as(
            memory
        )  # [tgt_size, tgt_size]
        output = self.decoder(tgt, memory, tgt_mask)  # [N, tgt_size, d_model]
        output = self.fc(output)  # [N, tgt_size, num_classes]
        return output

    def sample(self, image: Tensor) -> Tensor:
        """ Generate Formula.

        Args:
            image (Tensor): input image in gray scale, shape = [N, C=1, H, W].

        Returns:
            Tensor: predicted formula.
        """
        batch_size = image.shape[0]
        max_len = self.max_output_len
        feature = self.encode(image)
        output = (
            torch.full((batch_size, max_len), self.ignore_indices["pad"])
            .type_as(image)
            .long()
        )
        output[:, 0] = self.ignore_indices["sos"]
        for tgt_size in range(1, max_len):
            indices = output[:, :tgt_size]  # [B, tgt_size]
            logits = self.decode(indices, feature)  # [B, tgt_size, num_classes]
            tokens = torch.argmax(logits, dim=-1)  # [B, tgt_size]
            output[:, tgt_size] = tokens[:, -1]

        ends = np.argmax(output.cpu() == self.ignore_indices["pad"], axis=1)
        for i in range(batch_size):
            output[i, ends[i].item() :] = self.ignore_indices["pad"]
        return output.type_as(image)
