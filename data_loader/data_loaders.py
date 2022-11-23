import os
import pickle
import re

import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from tqdm.auto import tqdm

from utils import utils


class Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        if len(self.labels) == 0:
            item = [val[idx].clone().detach() for _, val in self.pair_dataset.items()]
        else:
            item = [val[idx].clone().detach() for _, val in self.pair_dataset.items()]
            item.append(self.labels[idx])
        return item

    def __len__(self):
        return len(self.pair_dataset["input_ids"])


class Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        train_ratio,
        shuffle,
        train_path,
        test_path,
        predict_path,
        use_swap,
        use_add_token=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.shuffle = shuffle

        self.train_path = train_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        model_list = {
            "bert": [
                "klue/roberta-small",
                "klue/roberta-base",
                "klue/roberta-large",
            ],
            "electra": [
                "monologg/koelectra-base-v3-discriminator",
                "monologg/koelectra-base-finetuned-sentiment",
            ],
            "roberta": [
                "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                "jhgan/ko-sroberta-multitask",
            ],
            "funnel": [
                "kykim/funnel-kor-base",
            ],
        }

        if model_name in model_list["bert"]:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        elif model_name in model_list["electra"]:
            self.tokenizer = transformers.ElectraTokenizer.from_pretrained(self.model_name)
        elif model_name in model_list["roberta"]:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
        elif model_name in model_list["funnel"]:
            self.tokenizer = transformers.FunnelTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        self.tokenizer.model_max_length = 256
        self.use_add_token = use_add_token
        if self.use_add_token:
            self.add_token = [
                "<PERSON>",
                "...",
                # "!!!",
                # "???",
                "ㅎㅎㅎ",
                "ㅋㅋㅋ",
                "ㄷㄷㄷ",
            ]
            self.new_token_count = self.tokenizer.add_tokens(self.add_token)
        else:
            self.new_token_count = 0
        self.swap = use_swap

        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence", "subject_entity", "object_entity"]

    def tokenizing(self, df):
        data = []
        sep_token = self.tokenizer.special_tokens_map["sep_token"]

        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = [e01 + sep_token + e02 for e01, e02 in zip(df["subject_entity"], df["object_entity"])]

        text = list(df["sentence"])
        if self.use_add_token:
            text = utils.text_preprocessing(text)

        data = self.tokenizer(
            concat_entity,
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return data

    def preprocessing(self, df):
        df = df.drop(columns=self.delete_columns)

        """기존 subject_entity와 subject entity를 word값으로만 대체"""
        subject_entity = []
        object_entity = []

        for sub, obj in zip(df["subject_entity"], df["object_entity"]):
            sub = eval(sub)
            obj = eval(obj)

            subject_entity.append(sub["word"].replace("'", ""))
            object_entity.append(obj["word"].replace("'", ""))

        preprocessed_df = pd.DataFrame(
            {
                "sentence": df["sentence"],
                "subject_entity": subject_entity,
                "object_entity": object_entity,
                "label": df["label"],
            }
        )

        try:
            if preprocessed_df["label"].iloc[0] == 100:  # test_data인 경우
                targets = []
            else:
                targets = preprocessed_df["label"].values.tolist()
                targets = utils.label_to_num(targets)
        except:
            targets = []

        inputs = self.tokenizing(preprocessed_df)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

            split = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=1004)
            for train_idx, val_idx in split.split(total_data, total_data["label"]):
                train_data = total_data.loc[train_idx]
                val_data = total_data.loc[val_idx]

            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)
            print("train data len : ", len(train_inputs["input_ids"]))
            print("valid data len : ", len(val_inputs["input_ids"]))

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            print("predict data len : ", len(predict_inputs["input_ids"]))

            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

    def new_vocab_size(self):
        return self.new_token_count + self.tokenizer.vocab_size


class KfoldDataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        k,
        num_splits,
        train_path,
        test_path,
        predict_path,
        use_swap,
        use_preprocessing=False,
    ):

        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.k = k
        self.num_splits = num_splits
        self.split_seed = 1204

        self.train_path = train_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        model_list = {
            "bert": [
                "klue/roberta-small",
                "klue/roberta-base",
                "klue/roberta-large",
            ],
            "electra": [
                "monologg/koelectra-base-v3-discriminator",
                "monologg/koelectra-base-finetuned-sentiment",
            ],
            "roberta": [
                "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                "jhgan/ko-sroberta-multitask",
            ],
            "funnel": ["kykim/funnel-kor-base"],
        }

        if model_name in model_list["bert"]:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        elif model_name in model_list["electra"]:
            self.tokenizer = transformers.ElectraTokenizer.from_pretrained(self.model_name)
        elif model_name in model_list["roberta"]:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
        elif model_name in model_list["funnel"]:
            self.tokenizer = transformers.FunnelTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        self.tokenizer.model_max_length = 128
        self.use_preprocessing = use_preprocessing
        if self.use_preprocessing:
            self.add_token = [
                "<PERSON>",
                "...",
                # "!!!",
                # "???",
                "ㅎㅎㅎ",
                "ㅋㅋㅋ",
                "ㄷㄷㄷ",
            ]
        else:
            self.add_token = [
                "<PERSON>",
            ]

        self.new_token_count = self.tokenizer.add_tokens(self.add_token)
        self.swap = use_swap

        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataframe, swap):
        data = []
        sep_token = self.tokenizer.special_tokens_map["sep_token"]
        print("ToKenizer info: \n", self.tokenizer)
        for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
            text = sep_token.join([item[text_column] for text_column in self.text_columns])
            if self.use_preprocessing:
                text = utils.text_preprocessing(text)
            outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
            data.append(outputs["input_ids"])

        if swap:
            for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
                text = sep_token.join([item[text_column] for text_column in self.text_columns[::-1]])
                if self.use_preprocessing:
                    text = utils.text_preprocessing(text)
                outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
                data.append(outputs["input_ids"])

        return data

    def preprocessing(self, data, swap):
        data = data.drop(columns=self.delete_columns)

        try:
            if swap:
                targets = data[self.target_columns].values.tolist() + data[self.target_columns].values.tolist()
            else:
                targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data, swap)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

            kf = KFold(
                n_splits=self.num_splits,
                shuffle=self.shuffle,
                random_state=self.split_seed,
            )
            all_splits = [d_i for d_i in kf.split(total_data)]
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            print("Number of splits: \n", self.num_splits)
            print("Before Swap Train data len: \n", len(train_indexes))
            print("Before Swap Valid data len: \n", len(val_indexes))

            train_inputs, train_targets = self.preprocessing(total_data.loc[train_indexes], self.swap)
            valid_inputs, valid_targets = self.preprocessing(total_data.loc[val_indexes], False)

            train_dataset = Dataset(train_inputs, train_targets)
            valid_dataset = Dataset(valid_inputs, valid_targets)

            print("After Swap Train data len: \n", len(train_inputs))
            print("After Swap Valid data len: \n", len(valid_inputs))

            self.train_dataset = train_dataset
            self.val_dataset = valid_dataset

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets = self.preprocessing(test_data, False)
            predict_inputs, predict_targets = self.preprocessing(predict_data, False)

            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

    def new_vocab_size(self):
        return self.new_token_count + self.tokenizer.vocab_size
