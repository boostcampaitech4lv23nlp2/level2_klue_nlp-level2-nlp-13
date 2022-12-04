import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from utils import utils


def inference(args, config):
    trainer = pl.Trainer(gpus=1, max_epochs=config.train.max_epoch, log_every_n_steps=1, deterministic=True)
    dataloader, model = utils.new_instance(config)
    model = utils.load_pretrained(model, config) 

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
    path = args.saved_model if args.saved_model is not None else config.path.best_model_path
    run_name = config.model.name + path.split("/")[-1]
    run_name = run_name.replace("/", "-")
    output.to_csv(f"./prediction/submission_{run_name}.csv", index=False)
