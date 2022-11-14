import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataloader.dataset import RE_Dataset, load_test_data, num_to_label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []

    for data in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                token_type_ids=data["token_type_ids"].to(device),
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Load Tokenizer ###
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    ### Load Model ###
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.best_model_path
    )
    model.to(device)

    ### Load Dataset ###
    test_id, tokenized_test, test_label = load_test_data(
        config.path.test_path, tokenizer
    )
    RE_test_dataset = RE_Dataset(tokenized_test, test_label)

    ### Predict ###
    pred_answer, output_prob = inference(model, RE_test_dataset, device)
    pred_answer = num_to_label(pred_answer)

    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    if not os.path.isdir("prediction"):
        os.mkdir("prediction")
    output.to_csv(
        "./prediction/submission.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장


if __name__ == "__main__":
    # config 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./configs/{args.config}.yaml")

    # seed 설정
    seed_everything(config.train.seed)

    main(config)
