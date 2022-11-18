import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def nll_loss(output, target):
    loss_func = nn.NLLLoss()
    return loss_func(output, target)


def L1_loss(output, target):
    loss_func = nn.L1Loss()
    return loss_func(output, target)


def mse_loss(output, target):
    loss_func = nn.MSELoss()
    return loss_func(output, target)


def rmse_loss(output, target):
    loss_func = nn.MSELoss()
    return torch.sqrt(loss_func(output, target))


def BCEWithLogitsLoss(output, target):
    loss_func = nn.BCEWithLogitsLoss()
    return loss_func(output, target)


def CELoss(output, target):
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    return loss_func(output, target)


# fmt: off
def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation", "org:top_members/employees", "org:members", "org:product", 
        "per:title", "org:alternate_names", "per:employee_of", "org:place_of_headquarters", "per:product", 
        "org:number_of_employees/members", "per:children", "per:place_of_residence", "per:alternate_names",
        "per:other_family", "per:colleagues", "per:origin", "per:siblings", "per:spouse", "org:founded",
        "org:political/religious_affiliation", "org:member_of", "per:parents", "org:dissolved",
        "per:schools_attended", "per:date_of_death", "per:date_of_birth", "per:place_of_birth", 
        "per:place_of_death", "org:founded_by", "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    preds = preds.argmax(-1)
    """ gpu에 할당되어 있던 tensor를 cpu tensor로 변환시켜줘야 에러 발생 안함 """
    return sklearn.metrics.f1_score(labels.cpu(), preds.cpu(), average="micro", labels=label_indices) * 100.0
# fmt: on


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


loss_config = {
    "nll": nll_loss,
    "l1": L1_loss,
    "mse": mse_loss,
    "rmse": rmse_loss,
    "bce": BCEWithLogitsLoss,
    "ce": CELoss,
    "acc": accuracy_score,
    "f1": klue_re_micro_f1,
    "auprc": klue_re_auprc,
    "RE": compute_metrics,
}
