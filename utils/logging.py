 import wandb
import pytz
import datetime

from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning.loggers import Logger, WandbLogger

def init_logger(config):   
    init_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    wandb.init(
        entity=config.wandb.team_account_name,
        project=config.wandb.project_repo,
        name=f"{config.wandb.name}_{config.wandb.info}_{init_time}",
    )
    wandb_logger = WandbLogger() # log_model="all"
    save_path = Path(config.path.save_path) / config.model.name / f"me{config.train.max_epoch}_bs{config.train.batch_size}_{wandb_logger.experiment.name}" 
    wandb_logger.experiment.config.update({"save_dir": save_path})
    return wandb_logger

class Logging(Logger):
    def __init__(self):
        super().__init__()

    @classmethod
    def init_logger(cls, config):
        init_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
        wandb.init(
            entity=config.wandb.team_account_name,
            project=config.wandb.project_repo,
            name=f"{config.wandb.name}_{config.wandb.info}_{init_time}",
        )
        wandb_logger = WandbLogger() # log_model="all"
        save_path = Path(config.path.save_path)
        cls.save_path = save_path / config.model.name / f"me{config.train.max_epoch}_bs{config.train.batch_size}_{wandb_logger.experiment.name}" 
        wandb_logger.experiment.config.update({"save_dir": cls.save_path})
        return wandb_logger

    @classmethod
    def save_config(cls, config):
        cls.config = config
        try:
            OmegaConf.save(cls.config, cls.save_path / "config.yaml")
        except FileNotFoundError as E:
            print("Training has been resumed but there is nothing new to log.")

