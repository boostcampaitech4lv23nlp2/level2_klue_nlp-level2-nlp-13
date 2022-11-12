import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import transformers
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR

from . import loss as loss_module


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name,
        lr,
        loss,
        new_vocab_size,
        use_freeze,
    ):  # 새로운 vocab 사이즈 설정
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=1,
        )

        if use_freeze == True:
            self.freeze()
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정
        self.loss_func = loss_module.loss_config[loss]

    def freeze(self):  # 추후 레이어를 반복하면서 얼리고 풀고 할 수 있게 훈련
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
        input_ids, attention_mask, token_type_ids = x
        x = self.plm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids, labels)
        loss = self.loss_func(logits, labels.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids, labels)
        loss = self.loss_func(logits, labels.float())
        self.log("val_loss", loss)
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), labels.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids, labels)
        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), labels.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids, labels)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer


class Klue_CustomModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        lr,
        loss,
        new_vocab_size,
        use_freeze,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.classifier_input = 1024
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(  # 기존 모델
            pretrained_model_name_or_path=model_name,
            num_labels=self.classifier_input,
        )
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정

        self.loss_func = loss_module.loss_config[loss]

        self.MLP_HEAD = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.classifier_input, 1),
        )

        if use_freeze == True:
            self.freeze()
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정
        self.loss_func = loss_module.loss_config[loss]

    def freeze(self):  # 추후 레이어를 반복하면서 얼리고 풀고 할 수 있게 훈련
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
        x = self.plm(x)["logits"]
        x = self.MLP_HEAD(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer


class Funnel_CustomModel(pl.LightningModule):  # 스케줄러 사용
    def __init__(
        self,
        model_name,
        lr,
        loss,
        new_vocab_size,
        use_freeze,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.plm = transformers.FunnelModel.from_pretrained(  # 기존 모델
            pretrained_model_name_or_path=model_name,
        )
        self.input_dim = transformers.FunnelConfig.from_pretrained(
            self.model_name,
        ).d_model  # 히든 벡터 차원

        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정

        self.loss_func = loss_module.loss_config[loss]

        self.Head = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.Tanh(),
            nn.Dropout(0.2),
        )
        self.Head2 = nn.Sequential(nn.Linear(1792, 1))

        if use_freeze == True:
            self.freeze()
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정
        self.loss_func = loss_module.loss_config[loss]

    def freeze(self):  # 추후 레이어를 반복하면서 얼리고 풀고 할 수 있게 훈련
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
        x = self.plm(x)[0]
        x = x[:, 0, :]  # x: 768
        y = self.Head(x)  # y: 1024
        x = torch.cat((x, y), dim=1)
        x = self.Head2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.95)  # 지수적으로 감소하게 해둠
        return [optimizer], [scheduler]


class Xlm_CustomModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        lr,
        loss,
        new_vocab_size,
        use_freeze,
    ):  # 새로운 vocab 사이즈 설정
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=1,
        )

        if use_freeze == True:
            self.freeze()
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정
        self.loss_func = loss_module.loss_config[loss]

    def freeze(self):  # 추후 레이어를 반복하면서 얼리고 풀고 할 수 있게 훈련
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
        x = self.plm(x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        # scheduler = LambdaLR(optimizer, lr_lambda=triangle_func)
        return [optimizer], [scheduler]


# def triangle_func(epoch):
#     max_lr_epoch = 50  # 삼각형의 꼭짓점
#     grad = 1 / max_lr_epoch  # 기울기
#     if max_lr_epoch > epoch:
#         return grad * epoch
#     else:
#         return max(0, 1 - grad * (epoch - max_lr_epoch))
