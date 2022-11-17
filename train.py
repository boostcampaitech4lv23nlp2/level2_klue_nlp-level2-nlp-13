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
    Trainer,
    TrainingArguments,
)
import wandb

from dataloader.dataset import load_train_dev_data, RE_Dataset, RE_Collator
from trainer.metrics import compute_metrics


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
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenized_train, train_label = load_train_dev_data(
        config.path.train_path
    )
    tokenized_dev, dev_label = load_train_dev_data(config.path.dev_path)

    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    RE_collator = RE_Collator(tokenizer)

    print("\033[38;2;31;169;250m" + "get model" + "\033[0m")
    model_config = AutoConfig.from_pretrained(config.model.name)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name, config=model_config
    )
    model.parameters
    model.to(device)

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
    )

    wandb.init(
        entity=config.wandb.team_account_name,
        project=config.wandb.project_repo,
        name=training_args.run_name,
    )
    wandb.config.update(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset,
        eval_dataset=RE_dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=RE_collator
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
