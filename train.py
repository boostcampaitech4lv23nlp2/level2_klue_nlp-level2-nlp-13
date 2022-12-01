import argparse
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
)
import wandb

from dataloader.dataset import load_train_dev_data, RE_Dataset, RE_Collator
from trainer.trainer import CustomTrainer
from trainer.metrics import compute_metrics
from trainer.optimizer import get_optimizer, get_scheduler
from data.utils.entity_marker import add_special_tokens
from models.entity_embeddings import CustomRobertaEmbeddings
from models.get_model import get_model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : ", device)

    print("\033[38;2;31;169;250m" + "get dataset" + "\033[0m")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, add_special_tokens=True
    )
    # Entity Marker를 적용할 경우 tokenizer에 special token 추가
    added_token_num = 0
    if config.data.entity_marker_type is not None:
        added_token_num, tokenizer = add_special_tokens(
            config.data.entity_marker_type, tokenizer
        )

    tokenized_train, train_label = load_train_dev_data(config.path.train_path)
    tokenized_dev, dev_label = load_train_dev_data(config.path.dev_path)

    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    RE_collator = RE_Collator(tokenizer)

    print("\033[38;2;31;169;250m" + "get model" + "\033[0m")
    model = get_model(config, tokenizer, added_token_num)

    model.parameters
    model.to(device)

    print(model) # ERASE LATER🥺

    print("\033[38;2;31;169;250m" + "get trainer" + "\033[0m")

    training_args = TrainingArguments(
        output_dir=config.train.checkpoints_dir,
        save_total_limit=config.train.save_total_limits,
        save_steps=config.train.save_steps,
        num_train_epochs=config.train.num_train_epochs,
        learning_rate=config.train.learning_rate,
        per_device_train_batch_size=config.train.train_batch_size,
        per_device_eval_batch_size=config.train.eval_batch_size,
        warmup_steps=config.train.warmup_steps,
        weight_decay=config.train.weight_decay,
        logging_dir=config.train.logging_dir,
        logging_steps=config.train.logging_steps,
        evaluation_strategy=config.train.evaluation_strategy,
        eval_steps=config.train.eval_steps,
        load_best_model_at_end=config.train.load_best_model_at_end,
        report_to="wandb",
        run_name=f"{config.wandb.name}_{config.wandb.info}",
        fp16=True,
        fp16_opt_level="01",
    )

    wandb.init(
        entity=config.wandb.team_account_name,
        project=config.wandb.project_repo,
        name=training_args.run_name,
    )
    wandb.config.update(training_args)

    trainer = CustomTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset,
        eval_dataset=RE_dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=RE_collator,
        # optimizers=optimizers, # FIX: valid 학습이 안됨 어디가 잘못되었는지 확인필요
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.train.early_stopping_patience
            )
        ],
    )

    print("\033[38;2;31;169;250m" + "Training start" + "\033[0m")
    trainer.train()
    model.save_pretrained("./best_model")


if __name__ == "__main__":
    # config 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./configs/{args.config}.yaml")

    # seed 설정
    seed_everything(config.train.seed)

    main(config)
