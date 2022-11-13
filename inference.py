import pandas as pd
import pytorch_lightning as pl
import torch

import train
from data_loader.data_loaders import Dataloader


def inference(args, conf):
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
    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)

    model, _, __ = train.load_model(args, conf, dataloader)

    model.eval()

    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )
    trainer.test(model=model, datamodule=dataloader)

    predictions = list(float(i) for i in torch.cat(predictions))
    predictions_b = list(round(float(i), 1) for i in predictions)
    predictions_n = [round(5 * x / (max(predictions) - min(predictions) + 1e-8), 1) for x in predictions]

    output = pd.read_csv("../data/sample_submission.csv")
    output_b = pd.read_csv("../data/sample_submission.csv")
    output_n = pd.read_csv("../data/sample_submission.csv")

    output["target"] = predictions
    output_b["target"] = predictions_b
    output_n["target"] = predictions_n

    output.to_csv("output.csv", index=False)
    output_b.to_csv("output_b.csv", index=False)
    output_n.to_csv("output_n.csv", index=False)
