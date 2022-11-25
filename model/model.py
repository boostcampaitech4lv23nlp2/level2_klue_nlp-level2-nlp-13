import warnings
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import transformers
import model.loss as loss_module
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR


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
        input_ids, token_type_ids, attention_mask = x
        x = self.plm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = self((input_ids, token_type_ids, attention_mask))

        loss = self.loss_func(logits, labels.long(), self.config)
        self.log("train_loss", loss, on_step=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("train_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("train_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("train_acc", metrics["accuracy"], on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = self((input_ids, token_type_ids, attention_mask))

        loss = self.loss_func(logits, labels.long(), self.config)
        self.log("val_loss", loss, on_step=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("val_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("val_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("val_acc", metrics["accuracy"], on_step=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = self((input_ids, token_type_ids, attention_mask))

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("test_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("test_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("test_acc", metrics["accuracy"], on_step=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, _ = batch
        logits = self((input_ids, token_type_ids, attention_mask))

        self.output_pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        self.output_prob = nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        return (self.output_pred, self.output_prob)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer


class CustomModel(BaseModel):
    def __init__(self, conf, new_vocab_size):
        super().__init__(conf, new_vocab_size)
