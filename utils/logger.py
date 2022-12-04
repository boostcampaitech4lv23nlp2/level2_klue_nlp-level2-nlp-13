import os
import wandb
import pytz
import datetime
import pathlib import Path

from omegaconf import OmegaConf


def log_config_yaml(config, save_path):
    try:
        OmegaConf.save(config, os.path.join(save_path, "config.yaml"))
    except FileNotFoundError as E:
        print("Training has been resumed but there is nothing new to log.")

def init_logger(config):   
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    wandb.init(
        entity=config.wandb.team_account_name,
        project=config.wandb.project_repo,
        name=f"{config.wandb.name}_{config.wandb.info}_{now_time}",
    )
    wandb_logger = WandbLogger() # log_model="all"
    save_path = Path(config.path.save_path) / f"{config.model.name}_me{config.train.max_epoch}_bs{config.train.batch_size}_{wandb_logger.experiment.name}" 
    wandb_logger.experiment.config.update({"save_dir": save_path})
    return wandb_logger
