import create_instance
import model.model as module_arch
import pytorch_lightning as pl
import torch
import utils.utils as utils
import wandb
from data_loader.data_loaders import Dataloader, KfoldDataloader
from pytorch_lightning.loggers import WandbLogger


def train(args, conf):
    wandb.init(
        entity=conf.wandb.team_account_name,
        project=conf.wandb.project_repo,
        name=f'{conf.wandb.name}_{conf.wandb.info}',
    )
    dataloader, model = create_instance.new_instance(conf)
    wandb_logger = WandbLogger()

    save_path = f"{conf.path.save_path}{conf.model.model_name}_maxEpoch{conf.train.max_epoch}_batchSize{conf.train.batch_size}_{wandb_logger.experiment.name}/"
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
                filename="{epoch}-{step}-{val_loss}",
            ),
        ],
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    wandb.finish()

    trainer.save_checkpoint(save_path + "model.ckpt")
    model.plm.save_pretrained(save_path)
    # torch.save(model, save_path + "model.pt")


def continue_train(args, conf):
    dataloader, model = create_instance.new_instance(conf)
    model, args, conf = create_instance.load_model(args, conf, dataloader, model)

    wandb_logger = WandbLogger(project=conf.wandb.project)
    save_path = f"{conf.path.save_path}{conf.model.model_name}_maxEpoch{conf.train.max_epoch}_batchSize{conf.train.batch_size}_{wandb_logger.experiment.name}/"
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
                filename="{epoch}-{step}-{val_loss}",
            ),
        ],
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    wandb.finish()

    trainer.save_checkpoint(save_path + "model.ckpt")
    # torch.save(model, save_path + "model.pt")


def k_train(args, conf):
    project_name = conf.wandb.project

    results = []
    num_folds = conf.k_fold.num_folds

    exp_name = WandbLogger(project=project_name).experiment.name
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

        if k + 1 == 1:
            name_ = f"{k+1}st_fold"
        elif k + 1 == 2:
            name_ = f"{k+1}nd_fold"
        elif k + 1 == 3:
            name_ = f"{k+1}rd_fold"
        else:
            name_ = f"{k+1}th_fold"
        wandb_logger = WandbLogger(project=project_name, name=exp_name + f"_{name_}")
        save_path = f"{conf.path.save_path}{conf.model.model_name}_maxEpoch{conf.train.max_epoch}_batchSize{conf.train.batch_size}_{wandb_logger.experiment.name}_{name_}/"
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
                    filename="{epoch}-{step}-{val_loss}",
                ),
            ],
        )

        trainer.fit(model=Kmodel, datamodule=k_datamodule)
        score = trainer.test(model=Kmodel, datamodule=k_datamodule)
        wandb.finish()

        results.extend(score)
        # torch.save(Kmodel, save_path + f"{name_} model.pt")
        trainer.save_checkpoint(save_path + f"{name_} model.ckpt")

    result = [x["test_pearson"] for x in results]
    score = sum(result) / num_folds
    print(f"{num_folds}-fold pearson 평균 점수: {score}")


def sweep(args, conf, exp_count):
    project_name = conf.wandb.project

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

        dataloader, model = create_instance.new_instance(conf, config=None)

        wandb_logger = WandbLogger(project=project_name)
        save_path = f"{conf.path.save_path}{conf.model.model_name}_sweep_id_{wandb.run.name}/"
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=conf.train.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
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
                    filename="{epoch}-{step}-{val_loss}",
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
