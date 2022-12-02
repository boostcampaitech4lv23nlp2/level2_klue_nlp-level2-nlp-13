import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from os import path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR
from torchmetrics.classification.accuracy import Accuracy
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

import model.loss as loss_module
import utils
from data_loader.data_loaders import BaseKFoldDataModule, KfoldDataloader

warnings.filterwarnings("ignore")


class BaseModel(pl.LightningModule):
    def __init__(self, config, new_vocab_size):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model_name = self.config.model.name
        self.lr = self.config.train.learning_rate
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=30,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        if self.config.train.use_frozen == True:
            self.freeze()
        self.plm.resize_token_embeddings(new_vocab_size)

        print(self.plm.__dict__)
        self.loss_func = loss_module.loss_config[self.config.train.loss]
        self.val_cm = config.train.print_val_cm
        self.test_cm = config.train.print_test_cm

        """variables to calculate inference loss"""
        self.output_pred = []
        self.output_prob = []
        """variables to calculate confusion matrix"""
        self.valid_preds = []
        self.valid_labels = []
        self.test_preds = []
        self.test_labels = []

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
        # input_ids, token_type_ids, attention_mask = x
        x = self.plm(**x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)

        loss = self.loss_func(logits, labels.long(), self.config)
        self.log("train_loss", loss, on_step=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("train_f1", metrics["micro f1 score"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_auprc", metrics["auprc"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_acc", metrics["accuracy"], on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)

        loss = self.loss_func(logits, labels.long(), self.config)
        self.log("val_loss", loss, on_step=self.config.utils.on_step, on_epoch=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("val_f1", metrics["micro f1 score"], on_step=self.config.utils.on_step, on_epoch=True, prog_bar=True)
        self.log("val_auprc", metrics["auprc"], on_step=self.config.utils.on_step, on_epoch=True, prog_bar=True)
        self.log("val_acc", metrics["accuracy"], on_step=self.config.utils.on_step, on_epoch=True, prog_bar=True)

        if len(self.valid_preds) == 0 and len(self.valid_labels) == 0:
            self.valid_preds = pred["predictions"]
            self.valid_labels = pred["label_ids"]
        else:
            self.valid_preds = np.concatenate((self.valid_preds, pred["predictions"]), axis=0)
            self.valid_labels = np.concatenate((self.valid_labels, pred["label_ids"]), axis=0)

        return loss

    def validation_epoch_end(self, outputs):
        if self.val_cm:
            utils.utils.get_confusion_matrix(self.valid_preds, self.valid_labels, "validation")

    def test_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)

        self.log(f"test_f1", metrics["micro f1 score"], on_step=self.config.utils.on_step, on_epoch=True, prog_bar=True)
        self.log(f"test_auprc", metrics["auprc"], on_step=self.config.utils.on_step, on_epoch=True, prog_bar=True)
        self.log(f"test_acc", metrics["accuracy"], on_step=self.config.utils.on_step, on_epoch=True, prog_bar=True)

        if len(self.test_preds) == 0 and len(self.test_labels) == 0:
            self.test_preds = pred["predictions"]
            self.test_labels = pred["label_ids"]
        else:
            self.test_preds = np.concatenate((self.test_preds, pred["predictions"]), axis=0)
            self.test_labels = np.concatenate((self.test_labels, pred["label_ids"]), axis=0)

    def test_epoch_end(self, outputs):
        if self.test_cm:
            utils.utils.get_confusion_matrix(self.test_preds, self.test_labels, "test")

    def predict_step(self, batch, batch_idx):
        tokens, _ = batch
        logits = self(tokens)

        self.output_pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        self.output_prob = F.softmax(logits, dim=-1).detach().cpu().numpy()

        return (self.output_pred, self.output_prob)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.config.train.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.config.train.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.config.train.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )
        elif self.config.train.scheduler == "LambdaLR":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.95**epoch
            )

        return [optimizer], [scheduler]


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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        tokens, _ = batch
        logits = torch.stack([m(tokens) for m in self.models]).mean(0)

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


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AuxiliaryClassificationRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        c1 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=2)
        c2 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        self.is_relation_classifier = RobertaClassificationHead(c1)  
        self.classifier = RobertaClassificationHead(c2)  

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        is_relation_logits = self.is_relation_classifier(sequence_output)  
        logits = self.classifier(sequence_output)  

        loss = 0 if labels is not None else None
        is_relation_output = (is_relation_logits,) + outputs[2:]  
        output = (logits,) + outputs[2:]  

        return is_relation_output, output


class AuxiliaryClassificationRobertaModel(BaseModel):
    def __init__(self, config, new_vocab_size):
        super().__init__(config, new_vocab_size)
        self.plm = AuxiliaryClassificationRobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="klue/roberta-large"
        )
        self.alpha = 0.2 # is_relation
        self.beta = 0.8
    
    def forward(self, x): 
        input_ids, attention_mask = x
        outputs= self.plm(input_ids=input_ids, attention_mask=attention_mask)
        is_relation_output, output = outputs[0][0], outputs[1][0]
        return is_relation_output, output


    def training_step(self, batch, batch_idx):
        tokens, labels, is_relation_labels = batch
        input_ids= tokens['input_ids'] 
        attention_mask =  tokens['attention_mask']
        
        is_relation_logits, logits = self((input_ids, attention_mask))

        is_relation_loss = self.loss_func(is_relation_logits, is_relation_labels.long(), self.config)
        loss = self.loss_func(logits, labels.long(), self.config)
        final_loss = self.alpha * is_relation_loss + self.beta * loss
        self.log("train_is_relation_loss", is_relation_loss, on_step=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_final_loss", final_loss, on_step=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("train_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("train_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("train_acc", metrics["accuracy"], on_step=True, prog_bar=True)

        return final_loss   

    def validation_step(self, batch, batch_idx): 
        tokens, labels, is_relation_labels  = batch
        input_ids= tokens['input_ids']
        attention_mask =  tokens['attention_mask']

        is_relation_logits, logits = self((input_ids, attention_mask))

        loss = self.loss_func(logits, labels.long(), self.config)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("val_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("val_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("val_acc", metrics["accuracy"], on_step=True, prog_bar=True)

        if len(self.valid_preds) == 0 and len(self.valid_labels) == 0:
            self.valid_preds = pred["predictions"]
            self.valid_labels = pred["label_ids"]
        else:
            self.valid_preds = np.concatenate((self.valid_preds, pred["predictions"]), axis=0)
            self.valid_labels = np.concatenate((self.valid_labels, pred["label_ids"]), axis=0)

        return loss   

    def test_step(self, batch, batch_idx):
        tokens, labels, is_relation_labels  = batch
        input_ids= tokens['input_ids']
        attention_mask =  tokens['attention_mask']
        is_relation_logits, logits = self((input_ids, attention_mask))

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("test_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("test_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("test_acc", metrics["accuracy"], on_step=True, prog_bar=True)

        if len(self.test_preds) == 0 and len(self.test_labels) == 0:
            self.test_preds = pred["predictions"]
            self.test_labels = pred["label_ids"]
        else:
            self.test_preds = np.concatenate((self.test_preds, pred["predictions"]), axis=0)
            self.test_labels = np.concatenate((self.test_labels, pred["label_ids"]), axis=0)

    def predict_step(self, batch, batch_idx):
        tokens, _ , _ = batch
        input_ids= tokens['input_ids']
        attention_mask =  tokens['attention_mask']
        is_relation_logits, logits = self((input_ids, attention_mask))

        self.output_pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        self.output_prob = nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        return (self.output_pred, self.output_prob)