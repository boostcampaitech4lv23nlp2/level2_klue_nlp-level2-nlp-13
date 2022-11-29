import os

from omegaconf import OmegaConf


def log_config_yaml(config, save_path):
    try:
        OmegaConf.save(config, os.path.join(save_path, "config.yaml"))
    except FileNotFoundError as E:
        print("Training has been resumed but there is nothing new to log.")
