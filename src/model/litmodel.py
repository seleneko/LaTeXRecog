""" LITMODEL.PY
"""

from typing import Tuple

from pytorch_lightning import LightningModule
from torch import Tensor, nn

from ..utils import path, utils
from .metric import EditDistance
from .model import LSTMConfig, ResNetRNN, ResNetTransformer, TransformerConfig


class LitModel(LightningModule):  # pylint: disable=too-many-ancestors,abstract-method
    """ Lightning ResNet and RNN / ResNet and Transformer.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        use_cnnrnn: bool,
        d_model: int,
        t_config: TransformerConfig,
        embed_size: int,
        lstm_config: LSTMConfig,
        max_output_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # model
        tokenizer = utils.load_vocab(path.VOCAB)
        ignore_indices = {
            "sos": tokenizer.sos,
            "eos": tokenizer.eos,
            "pad": tokenizer.pad,
        }
        resnet_rnn = ResNetRNN(
            embed_size=embed_size,
            num_classes=len(tokenizer),
            lstm_config=lstm_config,
            max_output_len=max_output_len,
        )
        resnet_transformer = ResNetTransformer(
            d_model=d_model,
            num_classes=len(tokenizer),
            t_config=t_config,
            ignore_indices=ignore_indices,
            max_output_len=max_output_len,
        )
        self.model = resnet_rnn if use_cnnrnn else resnet_transformer

        # metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_cer = EditDistance()
        self.test_cer = EditDistance()

    def training_step(self, *args, **_kwargs) -> Tensor:
        return self._training_step(*args)

    def validation_step(self, *args, **_kwargs) -> None:
        return self._validation_step(*args)

    def test_step(self, *args, **_kwargs) -> None:
        return self._test_step(*args)

    def _training_step(self, batch: Tuple[Tensor, Tensor], _batch_idx: int) -> Tensor:
        imgs, tgts = batch
        logits = self.model(imgs, tgts[:, :-1])
        loss = self.loss_fn(logits, tgts[:, 1:])
        self.log("train/loss", loss)
        return loss

    def _validation_step(self, batch: Tuple[Tensor, Tensor], _batch_idx: int) -> None:
        imgs, tgts = batch
        logits = self.model(imgs, tgts[:, :-1])
        loss = self.loss_fn(logits, tgts[:, 1:])
        self.log("val/loss", loss)
        preds = self.model.sample(imgs)
        val_correct, val_match = self.val_cer(preds, tgts)
        self.log("val/correct", val_correct)
        self.log("val/match", val_match)

    def _test_step(self, batch: Tuple[Tensor, Tensor], _batch_idx: int) -> None:
        imgs, tgts = batch
        preds = self.model.sample(imgs)
        test_correct, test_match = self.test_cer(preds, tgts)
        self.log("test/correct", test_correct)
        self.log("test/match", test_match)
