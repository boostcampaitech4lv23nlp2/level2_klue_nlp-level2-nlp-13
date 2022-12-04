import datetime
import logging

import pytorch_lightning as pl
import pytz
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from data_loader.data_loaders import KfoldDataloader
from model import model as module_arch
from utils import logger, utils


def train(config):
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")
    wandb.init(
        entity=config.wandb.team_account_name,
        project=config.wandb.project_repo,
        name=f"{config.wandb.name}_{config.wandb.info}_{now_time}",
    )

    dataloader, model = utils.new_instance(config)
    assert config.k_fold.use_k_fold == isinstance(
        dataloader, KfoldDataloader
    ), "Check your config again: Make sure `k_fold.use_k_fold` is compatible with `dataloader.architecture`"

    wandb_logger = WandbLogger(log_model="all")
    save_path = f"{config.path.save_path}{config.model.name}_maxEpoch{config.train.max_epoch}_batchSize{config.train.batch_size}_{wandb_logger.experiment.name}/"
    wandb_logger.experiment.config.update({"save_dir": save_path})
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,
        precision=config.utils.precision,
        num_sanity_val_steps=int(config.k_fold.use_k_fold is not True),
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["monitor"],
                mode=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["mode"],
                patience=config.utils.patience,
            ),
            utils.best_save(
                save_path=save_path,
                top_k=config.utils.top_k,
                monitor=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["monitor"],
                mode=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["mode"],
                filename="{epoch}-{step}-{val_loss}-{val_f1}",
            ),
        ]
    )

    if config.k_fold.use_k_fold:
        if config.utils.on_step is False:
            assert config.utils.patience >= config.k_fold.num_folds, "The given 'config.utils.patience' is way too low."
        internal_fit_loop = trainer.fit_loop
        trainer.fit_loop = getattr(module_arch, "KFoldLoop")(config.k_fold.num_folds, export_path=save_path)
        trainer.fit_loop.connect(internal_fit_loop)
        trainer.fit(model=model, datamodule=dataloader, ckpt_path=config.path.resume_path)
    else:
        trainer.fit(model=model, datamodule=dataloader, ckpt_path=config.path.resume_path)
        trainer.test(model=model, datamodule=dataloader)  # K-fold CV runs test_step internally as part of fitting step

    wandb.finish()
    config["path"]["best_model_path"] = trainer.checkpoint_callback.best_model_path
    logger.log_config_yaml(config, save_path)

    # trainer.save_checkpoint(save_path + "model.ckpt")
    # model.plm.save_pretrained(save_path)
    # torch.save(model, save_path + "model.pt")


def sweep(config, exp_count):
    project_name = config.wandb.project

    sweep_config = {
        "method": "bayes",
        "parameters": {
            "lr": {
                "distribution": "uniform",
                "min": 1e-5,
                "max": 3e-5,
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 30,
            "s": 2,
        },
    }

    sweep_config["metric"] = {"name": "test_pearson", "goal": "maximize"}

    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config

        dataloader, model = utils.new_instance(config, config=None)

        wandb_logger = WandbLogger(project=project_name)
        save_path = f"{config.path.save_path}{config.model.name}_sweep_id_{wandb.run.name}/"
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=config.train.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
            deterministic=True,
            precision=config.utils.precision,
            callbacks=[
                utils.early_stop(
                    monitor=utils.monitor_config[config.utils.monitor]["monitor"],
                    patience=config.utils.patience,
                    mode=utils.monitor_config[config.utils.monitor]["mode"],
                ),
                utils.best_save(
                    save_path=save_path,
                    top_k=config.utils.top_k,
                    monitor=utils.monitor_config[config.utils.monitor]["monitor"],
                    mode=utils.monitor_config[config.utils.monitor]["mode"],
                    filename="{epoch}-{step}-{val_loss}-{val_f1}",
                ),
            ],
        )
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        trainer.save_checkpoint(save_path + "model.ckpt")
        # torch.save(model, save_path + "model.pt")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name,
    )

    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=exp_count)
