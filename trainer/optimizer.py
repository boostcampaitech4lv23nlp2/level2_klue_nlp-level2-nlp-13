import torch
from transformers import AdamW


def get_optimizer(model, config):
    if config.train.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)

    return optimizer


def get_scheduler(optimizer, config):
    if config.train.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )  # TODO: 나중에 이 인자도 하이퍼파라미터 튜닝할 수 있게끔 수정하기

    elif config.train.scheduler == "LambdaLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 0.95**epoch
        )

    return scheduler
