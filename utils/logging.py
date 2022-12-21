import re
import wandb
import pytz
import datetime

from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger


class TemplateLogger:
    def __init__(self, config, logger, save_dir):
        self.config = config
        self.logger = logger
        self.save_dir = save_dir

    @classmethod
    def init_logger(cls, config):
        init_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
        
        wandb.init(
            entity=config.wandb.team_account_name,
            project=config.wandb.project_repo,
            name=f"{config.wandb.name}_{config.wandb.info}_{init_time}",
        )
        logger = WandbLogger() # log_model="all"
        save_dir = Path(config.path.save_path)
        save_dir = save_dir / config.model.name / f"me{config.train.max_epoch}_bs{config.train.batch_size}_{wandb_logger.experiment.name}" 
        logger.experiment.config.update({"save_dir": save_dir})
        
        return cls(config, logger, save_dir) 

    def save_config(self, config):
        if config.get("path.best_model_path", None):
            config.path.best_model_path = re.sub(r".+(?=saved_models)", "", config.path.best_model_path)           
            self.config = config
        try:
            OmegaConf.save(self.config, self.save_dir / "config.yaml")
        except FileNotFoundError as E:
            print("Such directory cannot be found. Training has been resumed but there is nothing new to log.")

