import datetime

import pytorch_lightning as pl
import pytz
import torch
from pytorch_lightning.loggers import WandbLogger

import utils
import model.model as module_arch
import utils.utils as utils
import wandb


def mhm_train(config):
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        entity=config.wandb.team_account_name,
        project=config.wandb.project_repo,
        name=f"{config.wandb.name}_{config.wandb.info}_{now_time}",
    )
    dataloader, model = utils.new_instance_mhm(config) # ⭐
    wandb_logger = WandbLogger()

    save_path = f"{config.path.save_path}{config.model.name}_maxEpoch{config.train.max_epoch}_batchSize{config.train.batch_size}_{wandb_logger.experiment.name}/"

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
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
    wandb.finish()

    trainer.save_checkpoint(f"{save_path}model.ckpt")
    model.plm.save_pretrained(save_path)
    # torch.save(model, save_path + "model.pt")