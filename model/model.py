import warnings
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import transformers
from transformers import AutoConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
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
        self.log("test_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("test_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("test_acc", metrics["accuracy"], on_step=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        tokens, _ = batch
        logits = self(tokens)

        self.output_pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        self.output_prob = nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        return (self.output_pred, self.output_prob)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer


class CustomModel(BaseModel):
    def __init__(self, conf, new_vocab_size):
        super().__init__(conf, new_vocab_size)


# ⭐⭐⭐MultipleHead⭐⭐⭐
class MultipleHeadRobertaModel(BaseModel):
    def __init__(self, conf, new_vocab_size):
        super().__init__(conf, new_vocab_size)
        self.model_config = AutoConfig.from_pretrained("klue/roberta-large")
        self.plm = CustomRobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="klue/roberta-large", 
            config=self.model_config)
        self.alpha = 0.5 # is_relation
        self.beta = 0.5
    
    def forward(self, x):
        input_ids, attention_mask = x
        outputs= self.plm(input_ids=input_ids, attention_mask=attention_mask)
        output_1, output_2 = outputs[0][0], outputs[1][0]
        return output_1, output_2
    
    def training_step(self, batch, batch_idx):
        input_ids, _, attention_mask, labels, is_relation_labels = batch
        logits_1, logits_2 = self((input_ids, attention_mask))

        is_relation_loss = self.loss_func(logits_1, is_relation_labels.long(), self.config)
        loss = self.loss_func(logits_2, labels.long(), self.config)
        final_loss = self.alpha * is_relation_loss + self.beta * loss
        self.log("train_is_relation_loss", is_relation_loss, on_step=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_final_loss", final_loss, on_step=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits_2.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("train_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("train_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("train_acc", metrics["accuracy"], on_step=True, prog_bar=True)

        return loss   

    def validation_step(self, batch, batch_idx):
        #input_ids, _ , attention_mask, labels, is_relation_labels = batch
        tokens, labels, is_relation_labels  = batch # 😰
        input_ids= tokens['input_ids'] # 😰
        attention_mask =  tokens['attention_mask'] # 😰
        logits_1, logits_2 = self((input_ids, attention_mask))

        is_relation_loss = self.loss_func(logits_1, is_relation_labels.long(), self.config)
        loss = self.loss_func(logits_2, labels.long(), self.config)
        final_loss = self.alpha * is_relation_loss + self.beta * loss
        self.log("val_is_relation_loss", is_relation_loss, on_step=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        self.log("val_final_loss", final_loss, on_step=True, prog_bar=True)

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits_2.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("val_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("val_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("val_acc", metrics["accuracy"], on_step=True, prog_bar=True)

        return loss   

    def test_step(self, batch, batch_idx):
        input_ids, _, attention_mask, labels, is_relation_labels = batch
        logits_1, logits_2 = self((input_ids, attention_mask))

        pred = {"label_ids": labels.detach().cpu().numpy(), "predictions": logits_2.detach().cpu().numpy()}
        metrics = loss_module.compute_metrics(pred)
        self.log("test_f1", metrics["micro f1 score"], on_step=True, prog_bar=True)
        self.log("test_auprc", metrics["auprc"], on_step=True, prog_bar=True)
        self.log("test_acc", metrics["accuracy"], on_step=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits_1, logits_2 = self((input_ids, attention_mask))

        self.output_pred = np.argmax(logits_2.detach().cpu().numpy(), axis=-1)
        self.output_prob = nn.functional.softmax(logits_2, dim=-1).detach().cpu().numpy()

        return (self.output_pred, self.output_prob)

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


class CustomRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        c1 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=2)
        c2 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        self.classifier_1 = RobertaClassificationHead(c1)  
        self.classifier_2 = RobertaClassificationHead(c2)  

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

        logits_1 = self.classifier_1(sequence_output)  
        logits_2 = self.classifier_2(sequence_output)  

        loss = 0 if labels is not None else None
        output_1 = (logits_1,) + outputs[2:]  
        output_2 = (logits_2,) + outputs[2:]  

        return output_1, output_2  