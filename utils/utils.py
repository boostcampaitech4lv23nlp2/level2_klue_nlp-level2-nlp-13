import pickle
import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_loader.data_loaders import Dataloader

def early_stop(monitor, patience, mode):
    early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode)
    return early_stop_callback


def best_save(save_path, top_k, monitor, mode, filename):
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=top_k,
        monitor=monitor,
        mode=mode,
        filename=filename,
    )
    return checkpoint_callback


def text_preprocessing(sentence):
    s = re.sub(r"!!+", "!!!", sentence)  # !한개 이상 -> !!! 고정
    s = re.sub(r"\?\?+", "???", s)  # ?한개 이상 -> ??? 고정
    s = re.sub(r"\.\.+", "...", s)  # .두개 이상 -> ... 고정
    s = re.sub(r"\~+", "~", s)  # ~한개 이상 -> ~ 고정
    s = re.sub(r"\;+", ";", s)  # ;한개 이상 -> ; 고정
    s = re.sub(r"ㅎㅎ+", "ㅎㅎㅎ", s)  # ㅎ두개 이상 -> ㅎㅎㅎ 고정
    s = re.sub(r"ㅋㅋ+", "ㅋㅋㅋ", s)  # ㅋ두개 이상 -> ㅋㅋㅋ 고정
    s = re.sub(r"ㄷㄷ+", "ㄷㄷㄷ", s)  # ㄷ두개 이상 -> ㄷㄷㄷ 고정
    return s

def new_instance(conf, config=None):

    if config is None:
        learning_rate = conf.train.learning_rate
    else:
        learning_rate = config.learning_rate

    dataloader = Dataloader(
        conf.model.model_name,
        conf.train.batch_size,
        conf.data.train_ratio,
        conf.data.shuffle,
        conf.path.train_path,
        conf.path.test_path,
        conf.path.predict_path,
        conf.data.swap,
    )

    model = module_arch.Model(
        conf.model.model_name,
        learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size,
        conf.train.use_frozen,
    )

    return dataloader, model


def load_model(args, conf, dataloader: Dataloader, model):
    '''
    불러온 모델이 저장되어 있는 디렉터리를 parsing함
    ex) 'save_models/klue/roberta-small_maxEpoch1_batchSize32_blooming-wind-57'
    '''
    save_path = "/".join(args.saved_model.split("/")[:-1])

    '''
    huggingface에 저장된 모델명을 parsing함
    ex) 'klue/roberta-small'
    '''
    model_name = "/".join(args.saved_model.split("/")[1:-1]).split("_")[0]

    if args.saved_model.split(".")[-1] == "ckpt":
        model = model.load_from_checkpoint(args.saved_model)
    elif args.saved_model.split(".")[-1] == "pt" and args.mode != "continue train" and args.mode != "ct":
        model = torch.load(args.saved_model)
    else:
        exit("saved_model 파일 오류")

    conf.path.save_path = save_path + "/"
    conf.model.model_name = "/".join(model_name.split("/")[1:])
    return model, args, conf


def label_to_num(label):
    with open("./data/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    num_label = dict_label_to_num[label]
    return num_label


def num_to_label(label):
    with open("./data/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    origin_label = [dict_num_to_label[v] for v in label]
    return origin_label


# 모니터링 할 쌍들
monitor_config = {
    "val_loss": {"monitor": "val_loss", "mode": "min"},
    "val_pearson": {"monitor": "val_pearson", "mode": "max"},
    "val_f1": {"monitor": "val_f1", "mode": "max"},
}


# def get_checkpoint_callback(criterion, save_frequency, prefix="checkpoint", use_modelcheckpoint_filename=False):

#     checkpoint_callback = None
#     if criterion == "step":
#         checkpoint_callback = CheckpointEveryNSteps(save_frequency, prefix, use_modelcheckpoint_filename)
#     elif criterion == "epoch":
#         checkpoint_callback = CheckpointEveryNEpochs(save_frequency, prefix, use_modelcheckpoint_filename)

#     return checkpoint_callback


# class CheckpointEveryNSteps(pl.Callback):
#     """
#     Save a checkpoint every N steps, instead of Lightning's default that checkpoints
#     based on validation loss.
#     """

#     def __init__(
#         self,
#         save_step_frequency,
#         prefix="checkpoint",
#         use_modelcheckpoint_filename=False,
#     ):
#         """
#         Args:
#             save_step_frequency: how often to save in steps
#             prefix: add a prefix to the name, only used if
#                 use_modelcheckpoint_filename=False
#             use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
#                 default filename, don't use ours.
#         """
#         self.save_step_frequency = save_step_frequency
#         self.prefix = prefix
#         self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

#     def on_batch_end(self, trainer: pl.Trainer, _):
#         """Check if we should save a checkpoint after every train batch"""
#         epoch = trainer.current_epoch
#         global_step = trainer.global_step
#         if global_step % self.save_step_frequency == 0:
#             if self.use_modelcheckpoint_filename:
#                 filename = trainer.checkpoint_callback.filename
#             else:
#                 filename = f"{self.prefix}_epoch={epoch}_global_step={global_step}.ckpt"
#             ckpt_path = os.path.join("model_save/", filename)
#             trainer.save_checkpoint(ckpt_path)


# class CheckpointEveryNEpochs(pl.Callback):
#     """
#     Save a checkpoint every N steps, instead of Lightning's default that checkpoints
#     based on validation loss.
#     """

#     def __init__(
#         self,
#         save_epoch_frequency,
#         prefix="checkpoint",
#         use_modelcheckpoint_filename=False,
#     ):
#         """
#         Args:
#             save_epoch_frequency: how often to save in epochs
#             prefix: add a prefix to the name, only used if
#                 use_modelcheckpoint_filename=False
#             use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
#                 default filename, don't use ours.
#         """
#         self.save_epoch_frequency = save_epoch_frequency
#         self.prefix = prefix
#         self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

#     def on_epoch_end(self, trainer: pl.Trainer, _):
#         """Check if we should save a checkpoint after every train epoch"""
#         epoch = trainer.current_epoch
#         global_step = trainer.global_step
#         if epoch % self.save_epoch_frequency == 0:
#             if self.use_modelcheckpoint_filename:
#                 filename = trainer.checkpoint_callback.filename
#             else:
#                 filename = f"{self.prefix}_epoch={epoch}_global_step={global_step}.ckpt"
#             ckpt_path = os.path.join("model_save/", filename)
#             trainer.save_checkpoint(ckpt_path)
