import os
import random
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

import create_instance
import model.model as module_arch
import utils.utils as utils
import wandb
from data_loader.data_loaders import Dataloader, KfoldDataloader

# fix random seeds for reproducibility


def new_instance_KLUE(conf):  # sweep 부분 때문에 두번째 인자 추가
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
    model = module_arch.Klue_CustomModel(
        conf.model.model_name,
        conf.train.learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size(),
        conf.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    return dataloader, model


def new_instance_XLM(conf):  # sweep 부분 때문에 두번째 인자 추가
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
    model = module_arch.Xlm_CustomModel(
        conf.model.model_name,
        conf.train.learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size(),
        conf.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    return dataloader, model


def new_instance_FUNNEL(conf):  # sweep 부분 때문에 두번째 인자 추가
    dataloader = Dataloader(
        conf.model.model_name,
        conf.train.batch_size,
        conf.data.train_ratio,
        conf.data.shuffle,
        conf.path.train_path,
        conf.path.test_path,
        conf.path.predict_path,
        conf.data.swap,
        True,
    )
    model = module_arch.Funnel_CustomModel(
        conf.model.model_name,
        conf.train.learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size(),
        conf.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    return dataloader, model


def full_model_step(conf, model_name, idx):
    SEED = conf.utils.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if idx == 0:
        dataloader, model = new_instance_FUNNEL(conf)
    elif idx == 1:
        dataloader, model = new_instance_KLUE(conf)
    elif idx == 2:
        dataloader, model = new_instance_XLM(conf)
    else:
        dataloader, model = new_instance_XLM(conf)
    project_name = conf.wandb.project

    wandb_logger = WandbLogger(project=project_name)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
    )
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    filename = re.sub("/", "_", conf.model.model_name)
    # trainer.save_checkpoint("./result/" + f"{model_name}.ckpt")
    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )
    predictions = list(float(i) for i in torch.cat(predictions))  # 리스트화

    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv(f"./result/output_{model_name}.csv", index=False)
    wandb.finish()


def K_model_step_train(conf):
    project_name = conf.wandb.project

    results = []
    num_folds = conf.k_fold.num_folds

    for k in range(num_folds):
        k_datamodule = KfoldDataloader(
            conf.model.model_name,
            conf.train.batch_size,
            conf.data.shuffle,
            k,
            conf.k_fold.num_split,
            conf.path.train_path,
            conf.path.test_path,
            conf.path.predict_path,
            conf.data.swap,
        )

        Kmodel = module_arch.Model(
            conf.model.model_name,
            conf.train.learning_rate,
            conf.train.loss,
            k_datamodule.new_vocab_size(),
            conf.train.use_frozen,
        )

        name_ = f"{k+1}th_fold"
        wandb_logger = WandbLogger(project=project_name, name=name_)
        save_path = f"{conf.path.save_path}{conf.model.model_name}_{conf.train.max_epoch}_{conf.train.batch_size}/"  # 모델 저장 디렉터리명에 wandb run name 추가
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=conf.train.max_epoch,
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[
                utils.early_stop(
                    monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                    patience=conf.utils.patience,
                    mode=utils.monitor_config[conf.utils.monitor]["mode"],
                ),
                utils.best_save(
                    save_path=save_path,
                    top_k=conf.utils.top_k,
                    monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                    mode=utils.monitor_config[conf.utils.monitor]["mode"],
                    filename=f"{k+1}_best_pearson_model",
                ),
            ],
        )

        trainer.fit(model=Kmodel, datamodule=k_datamodule)
        score = trainer.test(model=Kmodel, datamodule=k_datamodule)
        wandb.finish()

        results.extend(score)
        save_model = f"./result/kfold/{k}-fold"
        trainer.save_checkpoint(save_model + ".ckpt")

    result = [x["test_pearson"] for x in results]
    score = sum(result) / num_folds
    print(score)


def K_model_step_inference(conf):
    num_folds = conf.k_fold.num_folds
    for k in range(num_folds):

        trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)

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
            conf.train.learning_rate,
            conf.train.loss,
            dataloader.new_vocab_size(),
            conf.train.use_frozen,
        )  # 새롭게 추가한 토큰 사이즈 반영
        model = model.load_from_checkpoint(f"./result/kfold/{k}-fold.ckpt")
        model.eval()

        predictions = trainer.predict(
            model=model,
            datamodule=dataloader,
        )

        predictions = list(float(i) for i in torch.cat(predictions))  # 리스트화

        output = pd.read_csv("../data/sample_submission.csv")
        output["target"] = predictions
        output.to_csv(f"./result/kfold/{k}-fold.csv", index=False)


if __name__ == "__main__":
    if not os.path.exists("./result"):
        os.mkdir("./result")
    if not os.path.exists("./result/kfold"):
        os.mkdir("./result/kfold")

    # print("꼭 K-Fold 확인할 것(yaml)")
    config_name_list = ["funnel", "klue", "xlm", "xlm_5fold"]
    for idx, model_name in enumerate(config_name_list[0:3]):
        conf = OmegaConf.load(f"./config/{model_name}_ensemble.yaml")
        full_model_step(conf, model_name, idx)

    # k-fold 결과 내기
    conf = OmegaConf.load(f"./config/{config_name_list[3]}_ensemble.yaml")
    K_model_step_train(conf)  # 각 폴드 모델 만들기
    K_model_step_inference(conf)  # 각 csv 만들기
    # csv 합치기
    k_fold_value_list = []
    for k in range(conf.k_fold.num_folds):
        if k == 3:
            continue
        path = f"result/kfold/{k}-fold.csv"
        output = pd.read_csv(path)
        k_fold_value_list.append(output["target"])

    ## 최종 k_fold 결과 내기

    k_fold_mean_values = []
    for i in range(len(k_fold_value_list[0])):
        mean = 0
        for j in range(len(k_fold_value_list)):
            # print(value_list[j][i], end="    ")
            mean += k_fold_value_list[j][i]
        mean /= len(k_fold_value_list)
        # print(mean)
        k_fold_mean_values.append(mean)

    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = k_fold_mean_values
    output.to_csv("./result/output_xlm_5fold.csv", index=False)

    value_list = []
    for model_name in config_name_list:
        path = f"result/output_{model_name}.csv"

        output = pd.read_csv(path)
        print(output.head()["target"])
        value_list.append(output["target"])

    # print(len(value_list))
    # print(len(value_list[0]))

    mean_values = []
    for i in range(len(value_list[0])):
        mean = 0
        for j in range(len(value_list)):
            # print(value_list[j][i], end="    ")
            mean += value_list[j][i]
        mean /= len(value_list)
        # print(mean)
        mean_values.append(mean)
    # print(mean_values[0:10])

    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = mean_values
    output.to_csv("final_submit.csv", index=False)
