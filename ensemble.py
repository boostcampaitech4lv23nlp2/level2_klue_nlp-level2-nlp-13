import os
import re
import pytz
import datetime
import wandb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils import utils
from model.model import EnsembleVotingModel
from pytorch_lightning.loggers import WandbLogger
from sklearn.ensemble import RandomForestClassifier


def inference(args, config):
    assert config.ensemble.use_ensemble is True

    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")
    wandb.init(
        entity=config.wandb.team_account_name,
        project="Ensemble",
        name=f"{config.wandb.name}_{config.wandb.info}_{now_time}",
    )
    wandb_logger = WandbLogger()
    save_path = f"{config.path.save_path}ensemble/{wandb_logger.experiment.name}/"
    wandb_logger.experiment.config.update({"save_dir": save_path})

    trainer = pl.Trainer(gpus=1, max_epochs=config.train.max_epoch, log_every_n_steps=1, logger=wandb_logger)
    dataloader, model = utils.new_instance(config)
    ckpt_paths = [re.sub(r".+(?=saved_models)", "", path) for path in config.ensemble.ckpt_paths]

    model = EnsembleVotingModel(model, ckpt_paths)

    trainer.test(model=model, datamodule=dataloader)
    output = trainer.predict(model=model, datamodule=dataloader)

    pred_answer, output_prob = zip(*output)
    pred_answer = np.concatenate(pred_answer).tolist()
    output_prob = np.concatenate(output_prob, axis=0).tolist()
    pred_answer = utils.num_to_label(pred_answer)

    output = pd.DataFrame(
        {
            "id": range(len(pred_answer)),
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    if not os.path.isdir("prediction"):
        os.mkdir("prediction")

    output.to_csv(f"./prediction/submission_{wandb_logger.experiment.name}.csv", index=False)


def ensemble_csvs(paths):
    """
    `probs`은 평균을 내 계산하여 가장 높은 확률을 가진 pred_label을 return
    """
    assert len(paths) > 1, "There must be more than 1 path."

    dfs = [pd.read_csv(path) for path in paths]
    new_dfs = []
    for i, df in enumerate(dfs):
        if not _sanity_check(df):
            raise Exception(f"\u26A0 The following csv file fails the sanity check: {paths[i]} \u26A0")

        df["probs"] = df["probs"].apply(lambda row: np.array(eval(row)))  # convert to list of float
        # df[utils.num_to_label(np.arange(30))] = pd.DataFrame(df.probs.tolist(), index=df.index)
        new_dfs.append(df)
    total_df = pd.concat(new_dfs)
    new_total_df = pd.DataFrame()
    # total_df["probs"] = total_df.groupby("id")["probs"].apply("mean")

    new_total_df["probs"] = total_df.groupby("id")["probs"].apply("mean")
    max_indices = new_total_df["probs"].apply(np.argmax).to_list()
    new_total_df["pred_label"] = utils.num_to_label(max_indices)
    new_total_df["id"] = new_dfs[0].index
    new_total_df = new_total_df[["id", "pred_label", "probs"]]
    return new_total_df


def _sanity_check(df: pd.DataFrame):
    """
    리더보드가 제시하는 기준 만족하는지 체크
        1. csv의 column이 id, pred_label, probs로만 구성되어 있는지 확인
        2. probs 합계가 1인지 확인
        3. predicion label의 갯수가 30 개 미만이면 warning
        ?? 자릿수
    """
    ## whether the df has the same columns in order as supposed
    if not df.columns.to_list() == ["id", "pred_label", "probs"]:
        return False

    ## whether predicted probabilities sum up to 1
    def sums_up(row, eps=1e-5):
        probs = eval(row["probs"])
        return 1 if abs(1 - sum(probs)) < eps else 0

    if not all(df.apply(sums_up, axis=1)):
        return False

    ## count # of predicted labels
    pred_n = df.pred_label.nunique()
    if pred_n < 30:
        print(f"\u26A0 The number of prediction labels is {pred_n} \u26A0")

    return True


def run_random_forest(df: pd.DataFrame, sample_weight, max_depth=4):
    clf = RandomForestClassifier(max_depth=max_depth)
    clf.fit()
