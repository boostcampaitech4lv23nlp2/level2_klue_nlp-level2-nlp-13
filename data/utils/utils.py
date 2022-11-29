import os
import argparse
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf


def load_data(data_path):
    """데이터셋을 dataframe 형식으로 load
    Args:
        data_path (str): 불러올 데이터 파일 경로

    Returns:
        dataframe
    """
    raw_df = pd.read_csv(data_path)
    return raw_df


def add_special_tokens(marker_type, tokenizer):
    """토크나이저에 special tokens를 추가

    Args:
        marker_type (type): 적용하고자 하는 marker_type
        tokenizer (Tokenizer): 사용하는 토크나이저

    Returns:
        추가한 토큰 갯수, 토크나이저
    """
    if marker_type == "entity_marker":
        markers = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        added_token_num = tokenizer.add_special_tokens(
            {"additional_special_tokens": markers}
        )

    elif marker_type == "entity_marker_punc":
        added_token_num = 0

    elif marker_type == "typed_entity_marker":
        entity_types = ["PER", "ORG", "POH", "DAT", "LOC", "NOH"]
        markers = []

        for type in entity_types:
            markers.extend(
                (f"<S:{type}>", f"</S:{type}>", f"<O:{type}>", f"</O:{type}>")
            )
        added_token_num = tokenizer.add_special_tokens(
            {"additional_special_tokens": markers}
        )

    elif marker_type == "typed_entity_marker_punc_1":
        added_token_num = 0

    elif marker_type == "typed_entity_marker_punc_2":
        added_token_num = 0

    elif marker_type == "typed_entity_marker_punc_3":
        markers = ["PER", "ORG", "POH", "DAT", "LOC", "NOH"]
        added_token_num = tokenizer.add_special_tokens(
            {"additional_special_tokens": markers}
        )

    return added_token_num, tokenizer


def mark_entity(marker_type, word, word_type, is_subj):  # sourcery skip: switch
    """입력한 word에 marker_type에 따른 special token 추가

    Args:
        marker_type (str): 적용하고자 하는 marker_type
        word (str): entity 단어
        word_type (str): entity 단어의 유형
        is_subj (bool): subj entity이면 True, obj entity이면 False

    Returns:
        entity 단어에 marker type에 따른 special token로 감싼 문자열
    """
    if marker_type == "entity_marker":
        marked_word = f" [E1] {word} [/E1] " if is_subj else f" [E2] {word} [/E2] "

    elif marker_type == "entity_marker_punc":
        marked_word = f" @ {word} @ " if is_subj else f" # {word} # "

    elif marker_type == "typed_entity_marker":
        marked_word = (
            f" <S:{word_type}> {word} </S:{word_type}> "
            if is_subj
            else f" <O:{word_type}> {word} </O:{word_type}> "
        )

    elif marker_type == "typed_entity_marker_punc_1":
        entity_types = {
            "ORG": "기관",
            "PER": "인명",
            "POH": "기타",
            "DAT": "날짜",
            "LOC": "지명",
            "NOH": "수량",
        }
        marked_word = (
            f" @ * {entity_types[word_type]} * {word} @ "
            if is_subj
            else f" + ^ {entity_types[word_type]} ^ {word} + "
        )

    elif marker_type == "typed_entity_marker_punc_2":  # PER special token으로 추가하지 않고 사용
        marked_word = (
            f" @ * {word_type} * {word} @ "
            if is_subj
            else f" + ^ {word_type} ^ {word} + "
        )

    elif marker_type == "typed_entity_marker_punc_3":  # ORG special token으로 추가하고 사용
        marked_word = (
            f" @ * {word_type} * {word} @ "
            if is_subj
            else f" + ^ {word_type} ^ {word} + "
        )

    return marked_word


def get_entity_marked_dataframe(marker_type, df):

    """maker_type을 적용한 데이터프레임을 반환

    Args:
        marker_type (str): 적용하고자 하는 marker_type
        df (DataFrame): 원본 데이터프레임

    Returns:
        marker_type이 적용된 데이터프레임
    """
    assert marker_type in [
        "entity_marker",
        "entity_marker_punc",
        "typed_entity_marker",
        "typed_entity_marker_punc_1",
        "typed_entity_marker_punc_2",
        "typed_entity_marker_punc_3",
    ], "marker type은 [entity_marker, entity_marker_punc, typed_entity_marker, typed_entity_makrer_punc_1~3] 중에서 사용이 가능합니다."

    df_entity_marked = pd.DataFrame(
        columns=["id", "sentence", "subject_entity", "object_entity", "label", "source"]
    )

    for i in tqdm(range(len(df)), total=(len(df))):
        id, sentence, subj, obj, label, source = (
            df.iloc[i]["id"],
            df.iloc[i]["sentence"],
            df.iloc[i]["subject_entity"],
            df.iloc[i]["object_entity"],
            df.iloc[i]["label"],
            df.iloc[i]["source"],
        )

        subj_word, subj_start, subj_end, subj_type = eval(subj).values()
        obj_word, obj_start, obj_end, obj_type = eval(obj).values()

        try:
            # subject
            subj_start_idx = sentence.index(subj_word)
            subj_end_idx = subj_start_idx + len(subj_word) - 1

            # object
            obj_start_idx = sentence.index(obj_word)
            obj_end_idx = obj_start_idx + len(obj_word) - 1
        except Exception:
            print("Not Found")

        if subj_end_idx < obj_start_idx:
            sentence = (
                sentence[:subj_start_idx]
                + mark_entity(marker_type, subj_word, subj_type, is_subj=True)
                + sentence[subj_end_idx + 1 : obj_start_idx]
                + mark_entity(marker_type, obj_word, obj_type, is_subj=False)
                + sentence[obj_end_idx + 1 :]
            )
        elif obj_end_idx < subj_start_idx:
            sentence = (
                sentence[:obj_start_idx]
                + mark_entity(marker_type, obj_word, obj_type, is_subj=False)
                + sentence[obj_end_idx + 1 : subj_start_idx]
                + mark_entity(marker_type, subj_word, subj_type, is_subj=True)
                + sentence[subj_end_idx + 1 :]
            )

        df_entity_marked.loc[i] = [id, sentence, subj, obj, label, source]

    return df_entity_marked


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
    config = OmegaConf.load(f"./config/{args.config}.yaml")

    origin_train_path = "./data/raw_data/train.csv"
    origin_test_path = "./data/raw_data/test_data.csv"
    df_origin_train = load_data(origin_train_path)
    df_origin_test = load_data(origin_test_path)

    entity_marker_type = config.data_preprocess.marker_type

    df_preprocessed_train = get_entity_marked_dataframe(entity_marker_type, df_origin_train)
    df_preprocessed_test = get_entity_marked_dataframe(entity_marker_type, df_origin_test)

    if not os.path.exists("./data/preprocessed_data"):
        os.makedirs("./data/preprocessed_data")
    df_preprocessed_train.to_csv(f"./data/preprocessed_data/train.{entity_marker_type}.csv", index=False)
    df_preprocessed_test.to_csv(f"./data/preprocessed_data/test.{entity_marker_type}.csv", index=False)