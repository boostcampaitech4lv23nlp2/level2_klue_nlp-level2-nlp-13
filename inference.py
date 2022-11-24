import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from utils import utils


def inference(args, conf):
    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)
    dataloader, model = utils.new_instance(conf)
    model, _, __ = utils.load_model(args, conf, dataloader, model)

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
    output.to_csv("./prediction/submission.csv", index=False)
