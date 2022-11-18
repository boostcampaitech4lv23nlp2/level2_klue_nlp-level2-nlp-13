import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from transformers import Trainer
from trainer.loss_function import FocalLoss
import wandb


class CustomTrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        """
        Args:
            get_cm (bool, optional): wandb에서 confusion matrix를 그릴지 여부. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.get_cm = config.train.get_cm
        self.loss_fn = config.train.loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_fn == "FocalLoss":
            loss_fct = FocalLoss()

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            loss = loss_fct(outputs[0], labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, *args, **kwargs):
        eval_loop_output = super().evaluation_loop(*args, **kwargs)

        if self.get_cm:
            pred = eval_loop_output.predictions
            label_ids = eval_loop_output.label_ids

            self.get_confusion_matrix(pred, label_ids)
        return eval_loop_output

    def get_confusion_matrix(self, pred, label_ids):
        cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)

        cm_plot = sns.heatmap(cm, cmap="Blues", fmt="d", annot=True, ax=ax)
        cm_plot.set_xlabel("pred")
        cm_plot.set_ylabel("true")
        cm_plot.set_title("confusion matrix")

        wandb.log({"confusion_matrix": wandb.Image(fig)})
