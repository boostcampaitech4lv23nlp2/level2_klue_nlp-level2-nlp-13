import os
import argparse
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from entity_marker import load_data, get_entity_marked_dataframe
from preprocess import drop_duplicates, split_train_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="base_config")
    parser.add_argument("--mode", "-m", default="train")
    parser.add_argument(
        "--saved_model",
        "-s",
        default=None,
        help="저장된 모델의 파일 경로를 입력해주세요. 예시: saved_models/klue/roberta-small/epoch=?-step=?.ckpt 또는 save_models/model.pt",
    )
    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./configs/{args.config}.yaml")

    origin_train_path = "./data/raw_data/train.csv"
    origin_test_path = "./data/raw_data/test_data.csv"
    df_origin_train = load_data(origin_train_path)
    df_origin_test = load_data(origin_test_path)

    # 중복 제거
    df_origin_train_drop_duplicates = drop_duplicates(df_origin_train)

    # train → train valid 분리
    df_train, df_valid = split_train_valid(
        valid_ratio=0.2, dataframe=df_origin_train_drop_duplicates
    )

    entity_marker_type = "typed_entity_marker"  # [entity_marker, entity_marker_punc, typed_entity_marker, typed_entity_makrer_punc_1~3]
    df_preprocessed_train = get_entity_marked_dataframe(entity_marker_type, df_origin_train) # train 전체
    df_preprocessed_train_08 = get_entity_marked_dataframe(entity_marker_type, df_train) # train의 0.8
    df_preprocessed_valid_02 = get_entity_marked_dataframe(entity_marker_type, df_valid) # train의 0.2
    df_preprocessed_test = get_entity_marked_dataframe(entity_marker_type, df_origin_test)

    # csv로 저장
    if not os.path.exists("./data/preprocessed_data"):
        os.makedirs("./data/preprocessed_data")

    df_preprocessed_train.to_csv(f"./data/preprocessed_data/train.{entity_marker_type}.csv", index=False)
    df_preprocessed_train_08.to_csv(f"./data/preprocessed_data/train08.{entity_marker_type}.csv", index=False)
    df_preprocessed_valid_02.to_csv(f"./data/preprocessed_data/valid02.{entity_marker_type}.csv", index=False)
    df_preprocessed_test.to_csv(f"./data/preprocessed_data/test.{entity_marker_type}.csv", index=False)