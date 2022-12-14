import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils import utils
from model.model import EnsembleVotingModel
from data_loader.data_loaders import KfoldDataloader

def inference(args, config):
    trainer = pl.Trainer(gpus=1, max_epochs=config.train.max_epoch, log_every_n_steps=1)
    dataloader, model = utils.new_instance(config)
    # model, _, __ = utils.load_model(args, config, dataloader, model)
    model = EnsembleVotingModel(model, config.ensemble.ckpt_paths)

    output = trainer.predict(model=model, datamodule=dataloader) # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html
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
    output.to_csv("./prediction/submission.csv", index=False)
