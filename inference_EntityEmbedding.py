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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from dataloader.dataset import RE_Dataset, RE_Collator, load_test_data, num_to_label
from data.utils.entity_marker import add_special_tokens
from models.entity_embeddings import CustomRobertaEmbeddings
from models.get_model import get_model
from models.RobertaEmbeddingWithEntity import RobertaForSequenceClassificationWithEntityEmbedding

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inference(model, tokenizer, sentences, device):
    dataloader = DataLoader(
        sentences, batch_size=16, shuffle=False, collate_fn=RE_Collator(tokenizer)
    )
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
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, add_special_tokens=True
    )
    # Entity Marker를 적용할 경우
    if config.data.entity_marker_type is not None:
        added_token_num, tokenizer = add_special_tokens(
            config.data.entity_marker_type, tokenizer
        )

    ### Load Model ###
    # model = get_model(config, tokenizer, added_token_num)
    # model.load_state_dict(torch.load("./best_model/pytorch_model.bin"), strict=False)

    ## load my model
    MODEL_NAME = config.model.best_model_path # model dir.
    model = RobertaForSequenceClassificationWithEntityEmbedding.from_pretrained(MODEL_NAME)
    if config.data.entity_marker_type is not None:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    print(model)

    ### Load Dataset ###
    test_id, test_data, test_label = load_test_data(config.path.test_path)
    RE_test_dataset = RE_Dataset(test_data, test_label)

    ### Predict ###
    pred_answer, output_prob = inference(model, tokenizer, RE_test_dataset, device)
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