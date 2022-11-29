import os

from omegaconf import OmegaConf


def log_config_yaml(config, save_path):
    OmegaConf.save(config, os.path.join(save_path, "config.yaml"))

