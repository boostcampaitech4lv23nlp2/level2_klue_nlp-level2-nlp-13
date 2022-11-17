import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path):
    """데이터셋을 dataframe 형식으로 load
    Args:
        data_path (str): 불러올 데이터 파일 경로

    Returns:
        dataframe
    """
    raw_df = pd.read_csv(data_path)
    return raw_df


def drop_duplicates(dataframe):
    """데이터프레임내의 중복 제거
    Args:
        dataframe (DataFrame): 중복을 제거할 데이터프레임

    Returns:
        중복 제거된 dataframe
    """
    df_drop_duplicated = dataframe.drop_duplicates(
        subset=["sentence", "subject_entity", "object_entity"]
    ).reset_index(drop=True)
    return df_drop_duplicated


def split_train_valid(valid_ratio, dataframe):
    """데이터셋을 train과 valid set으로 class 비율을 맞춰서 분리
    Args:
        valid_ratio (float): valid set 비율
        dataframe (DataFrame) : 분리하고자 하는 데이터프레임

    Returns:
        train/valid로 분리된 데이터프레임
    """

    # df_x_data = raw_df.drop(["label"], axis=1)
    df_y_data = dataframe["label"]

    df_train, df_valid = train_test_split(
        dataframe, test_size=valid_ratio, random_state=42, stratify=df_y_data
    )
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    return df_train, df_valid


if __name__ == "__main__":
    # 데이터 로드
    train_path = "./data/raw_data/train.csv"
    df_origin = load_data(train_path)

    # 중복 제거
    df_drop_duplicates = drop_duplicates(df_origin)

    # train valid 분리
    df_train, df_valid = split_train_valid(
        valid_ratio=0.2, dataframe=df_drop_duplicates
    )

    # csv로 저장
    df_train.to_csv("./data/preprocessed_data/train.preprocessed.csv", index=False)
    df_valid.to_csv("./data/preprocessed_data/valid.preprocessed.csv", index=False)
