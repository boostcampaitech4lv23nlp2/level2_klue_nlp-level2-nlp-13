import warnings
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers
import model.loss as loss_module

from copy import deepcopy
from os import path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR
from torchmetrics.classification.accuracy import Accuracy
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn
from data_loader.data_loaders import KfoldDataloader, BaseKFoldDataModule


warnings.filterwarnings("ignore")

class BaseModel(pl.LightningModule):
    def __init__(self, config, new_vocab_size):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model_name = self.config.model.name
        self.lr = self.config.train.learning_rate
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=30,
        )

        if self.config.train.use_frozen == True:
            self.freeze()
        self.plm.resize_token_embeddings(new_vocab_size)
        self.loss_func = loss_module.loss_config[self.config.train.loss]

        """variables to calculate inference loss"""
        self.output_pred = []
        self.output_prob = []

    def freeze(self):
        for name, param in self.plm.named_parameters():
            param.requires_grad = False
            if name in [
                "classifier.dense.weight",
                "classifier.dense.bias",
                "classifier.out_proj.weight",
                "classifier.out_proj.bias",
            ]:
                param.requires_grad = True

    def forward(self, x):
        #input_ids, token_type_ids, attention_mask = x
        x = self.plm(**x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)

        loss = self.loss_func(logits, labels.long(), self.config)
        self.log("train_loss", loss, on_step=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("train_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("train_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("train_acc", metrics["accuracy"], on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)

        loss = self.loss_func(logits, labels.long(), self.config)
        self.log("val_loss", loss, on_step=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("val_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("val_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("val_acc", metrics["accuracy"], on_step=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        fold_idx = self.trainer.fit_loop.current_fold
        self.log(f"test_fold{fold_idx}_f1", metrics["micro f1 score"], on_step=False, prog_bar=True)
        self.log(f"test_fold{fold_idx}_auprc", metrics["auprc"], on_step=False, prog_bar=True)
        self.log(f"test_fold{fold_idx}_acc", metrics["accuracy"], on_step=False, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        tokens, _ = batch
        logits = self(tokens)

        self.output_pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        self.output_prob = F.softmax(logits, dim=-1).detach().cpu().numpy()

        return (self.output_pred, self.output_prob)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer


class CustomModel(BaseModel):
    def __init__(self, config, new_vocab_size):
        super().__init__(config, new_vocab_size)


class EnsembleVotingModel(pl.LightningModule):
    """Model for KFold CV"""
    def __init__(self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        # self.test_acc = Accuracy()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Compute the averaged predictions over the `num_folds` models.
        tokens, labels = batch
        logits = torch.stack([m(tokens) for m in self.models]).mean(0)
        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log(f"ensemble_f1", metrics["micro f1 score"])
        self.log(f"ensemble_auprc", metrics["auprc"])
        self.log(f"ensemble_acc_fold", metrics["accuracy"])

    def predict_step(self, batch, batch_idx, dataloader_idx):
        tokens, _ = batch
        logits = self(tokens)

        self.output_pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        self.output_prob = nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        return (self.output_pred, self.output_prob)


class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.

        # the test loop normally expects the model to be the pure LightningModule, but since we are running the
        # test loop during fitting, we need to temporarily unpack the wrapped module
        wrapped_model = self.trainer.strategy.model
        self.trainer.strategy.model = self.trainer.strategy.lightning_module
        self.trainer.test_loop.run()
        self.trainer.strategy.model = wrapped_model
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(path.join(self.export_path, f"fold_{self.current_fold}.ckpt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set. Run Ensemble.test_loop when there is a test-specific set"""
        checkpoint_paths = [path.join(self.export_path, f"fold_{f_idx + 1}.ckpt") for f_idx in range(self.num_folds)]
        if self.trainer.datamodule.train_path != self.trainer.datamodule.test_path:
            voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
            voting_model.trainer = self.trainer
            # This requires to connect the new model and move it the right device.
            self.trainer.strategy.connect(voting_model)
            self.trainer.strategy.model_to_device()
            self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
