import pickle

import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        
    def __getitem__(self, idx):
        sentence = self.dataset['sentence'][idx]
        subject_entity = self.dataset['subject_entity'][idx]
        object_entity = self.dataset['object_entity'][idx]
        label = self.labels[idx]
        return sentence, subject_entity, object_entity, label

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset):
    """기존 subject_entity와 subject entity를 word값으로만 대체

    Args:
        dataset (DataFrame): 원본 csv 파일을 읽은 데이터프레임

    Returns:
        원하는 형태의 DataFrame
    """
    subject_entity = []
    object_entity = []

    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        subject_entity.append(i)
        object_entity.append(j)

    return pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러옴

    Args:
        dataset_dir (str): 데이터 csv 파일 경로

    Returns:
        원하는 형태로 변경한 DataFrame
    """
    pd_dataset = pd.read_csv(dataset_dir)
    return preprocessing_dataset(pd_dataset)


def label_to_num(label):
    with open("./data/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)

    num_label = [dict_label_to_num[v] for v in label]
    return num_label


def num_to_label(label):
    """숫자로 되어있는 class를 원본 문자열 라벨로 변환"""
    with open("./data/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    origin_label = [dict_num_to_label[v] for v in label]
    return origin_label


def tokenized_dataset(sentences, tokenizer, subject_entities, object_entities):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = [
        e01 + "[SEP]" + e02
        for e01, e02 in zip(subject_entities, object_entities)
    ]

    tokenized_sentences = tokenizer(
        concat_entity,
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    return tokenized_sentences


def load_train_dev_data(dataset_dir):
    data = load_data(dataset_dir)
    label = label_to_num(data["label"].values)

    return data, label


def load_test_data(dataset_dir):
    test_data = load_data(dataset_dir)
    test_label = list(map(int, test_data["label"].values))

    return test_data["id"], test_data, test_label

class RE_Collator(object):
    '''
    tokenization을 적용, label을 tensor로 바꿔주는 data collator.
    '''
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sentences, subject_entities, object_entities, labels = zip(*batch)

        # tokenization
        tokenized_dict = tokenized_dataset(sentences, self.tokenizer, subject_entities, object_entities)
        
        item = {key: val for key, val in tokenized_dict.items()}
        item['labels'] = torch.tensor(labels)
        return item
        