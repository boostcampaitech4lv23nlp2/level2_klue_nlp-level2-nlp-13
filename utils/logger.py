import os
import wandb
import pytz
import datetime
import pathlib import Path

from omegaconf import OmegaConf
from pytorch_lightning.loggers import Logger

def save_config(config, save_path):
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

class Logger(Logger):
    def __init__(self):
        super().__init__()

    @classmethod
    def init_logger(cls, config):
        cls.config = config
        init_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
        save_path = Path(config.path.save_path)
        
        # if wandb
        wandb.init(
            entity=config.wandb.team_account_name,
            project=config.wandb.project_repo,
            name=f"{config.wandb.name}_{config.wandb.info}_{init_time}",
        )
        cls.logger = WandbLogger()
        cls.save_path = save_path / config.model.name / f"me{config.train.max_epoch}_bs{config.train.batch_size}_{wandb_logger.experiment.name}"
        cls.logger.experiment.config.update({"save_dir":cls.save_path})
        return wandb_logger
    
    @classmethod
    def save_config(self):
        try:
            OmegaConf.save(cls.config, cls.save_path, "config.yaml"))
        except FileNotFoundError as E:
            print("Training has been resumed but there is nothing new to log.")

