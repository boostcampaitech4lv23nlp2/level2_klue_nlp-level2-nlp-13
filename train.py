import logging
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from data_loader.data_loaders import KfoldDataloader
from model import model as module_arch
from utils import logger, utils


def train(config):
    logger = logger.init_logger(config)
    dataloader, model = utils.new_instance(config)
    assert config.k_fold.use_k_fold == isinstance(
        dataloader, KfoldDataloader
    ), "Check your config again: Make sure `k_fold.use_k_fold` is compatible with `dataloader.architecture`"
    monitor_configs = utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.max_epoch,
        log_every_n_steps=1,
        logger=logger,
        deterministic=True,
        precision=config.utils.precision,
        num_sanity_val_steps=int(config.k_fold.use_k_fold is not True),
        callbacks=[
            EarlyStopping(
                monitor=monitor_configs["monitor"],
                mode=monitor_configs["mode"],
                patience=config.utils.patience,
            ),
            ModelCheckpoint(
                save_path=save_path,
                save_top_k=config.utils.top_k,
                monitor=monitor_configs["monitor"],
                mode=monitor_configs["mode"],
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
