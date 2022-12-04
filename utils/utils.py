import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

import data_loader.data_loaders as datamodule_arch
import model.model as module_arch
import wandb
import pathlib
import re

def init_modules(config):
    dataloader = getattr(datamodule_arch, config.dataloader.architecture)(config)
    model = getattr(module_arch, config.model.architecture)(config, dataloader.new_vocab_size)

    return dataloader, model


def load_pretrained(model, config):
    """
    Load weights of a pretrained language model.
    """
    if path:= config.get("path.best_model_path", None):
       # if "all" mode
       model = model.load_from_checkpoint(path)
    else:
        # if "inference" mode
        path = config.path.ckpt_path
        pretrained_model = torch.load(path)
        if isinstance(pretrained_model, torch.nn.Module):
            model.plm = model.load_state_dict(path, strict=False)
            print(f"Replaced weights of {model.plm.__class__.__name__}")

    return model

def text_preprocessing(sentence):
    # s = re.sub(r"!!+", "!!!", sentence)  # !한개 이상 -> !!! 고정
    # s = re.sub(r"\?\?+", "???", s)  # ?한개 이상 -> ??? 고정
    # s = re.sub(r"\.\.+", "...", s)  # .두개 이상 -> ... 고정
    # s = re.sub(r"\~+", "~", s)  # ~한개 이상 -> ~ 고정
    # s = re.sub(r"\;+", ";", s)  # ;한개 이상 -> ; 고정
    # s = re.sub(r"ㅎㅎ+", "ㅎㅎㅎ", s)  # ㅎ두개 이상 -> ㅎㅎㅎ 고정
    # s = re.sub(r"ㅋㅋ+", "ㅋㅋㅋ", s)  # ㅋ두개 이상 -> ㅋㅋㅋ 고정
    # s = re.sub(r"ㄷㄷ+", "ㄷㄷㄷ", s)  # ㄷ두개 이상 -> ㄷㄷㄷ 고정
    return sentence


def label_to_num(label):
    with open("./data/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    num_label = [dict_label_to_num[v] for v in label]
    return num_label


def num_to_label(label):
    with open("./data/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    origin_label = [dict_num_to_label[v] for v in label]
    return origin_label


def get_confusion_matrix(pred, label_ids, mode=None):
    cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)

    cm_plot = sns.heatmap(cm, cmap="Blues", fmt="d", annot=True, ax=ax)
    cm_plot.set_xlabel("pred")
    cm_plot.set_ylabel("true")
    cm_plot.set_title(f"{mode} confusion matrix")

    wandb.log({f"{mode} confusion_matrix": wandb.Image(fig)})


def monitor_config(key, on_step):
    """Returns proper metric monitor-mode pair."""
    mapping = {
        "val_loss": {"monitor": "val_loss", "mode": "min"},
        "val_pearson": {"monitor": "val_pearson", "mode": "max"},
        "val_f1": {"monitor": "val_f1", "mode": "max"},
    }
    new_mapping = mapping.copy()
    if on_step is True:
        for m in mapping:
            for detail in ["step", "epoch"]:
                new_mapping[f"{m}_{detail}"] = mapping[m]
    else:
        if key.endswith("step"):
            raise ValueError(f"Cannot monitor {key} when on_step is set 'False'")

    return new_mapping[key]


