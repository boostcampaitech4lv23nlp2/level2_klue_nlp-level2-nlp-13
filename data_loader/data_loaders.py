import os
import pickle
import re
from abc import ABC, abstractmethod

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        sentence = self.df["sentence"].iloc[idx]
        subject_entity = self.df["subject_entity"].iloc[idx]
        object_entity = self.df["object_entity"].iloc[idx]
        label = self.df["label"].iloc[idx]
        return sentence, subject_entity, object_entity, label

    def __len__(self):
        return len(self.df)


class BaseDataloader(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.model_name = config.model.name
        self.batch_size = config.train.batch_size
        self.train_ratio = config.dataloader.train_ratio
        self.shuffle = config.dataloader.shuffle

        num_cpus = os.cpu_count()
        self.num_workers = num_cpus if self.batch_size // num_cpus <= num_cpus else int((self.batch_size // num_cpus) ** 0.5)

        self.train_path = config.path.train_path
        self.test_path = config.path.test_path
        self.predict_path = config.path.predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        assert isinstance(self.train_ratio, float) and self.train_ratio > 0.0 and self.train_ratio <= 1.0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = config.tokenizer.max_length
        self.new_tokens = list(config.tokenizer.new_tokens)
        self.use_syllable_tokenize = config.tokenizer.syllable

        self.new_token_count = 0
        if self.new_tokens != []:
            self.new_token_count += self.tokenizer.add_tokens(self.new_tokens)
            print(f"{self.new_token_count} new token(s) are added to the vocabulary.")

    def batchify(self, batch):
        """data collator"""
        sentences, subject_entities, object_entities, labels = zip(*batch)

        outs = self.tokenize(sentences, subject_entities, object_entities)
        labels = torch.tensor(labels)
        return outs, labels

    def tokenize(self, sentences, subject_entities, object_entities):
        """
        tokenizer로 과제에 따라 tokenize
        """
        sep_token = self.tokenizer.special_tokens_map["sep_token"]

        if self.use_syllable_tokenize:
            entities = [[e01, e02] for e01, e02 in zip(subject_entities, object_entities)]
            tokens = self.syllable_tokenizer(entities, sentences, self.max_length)
        else:
            concat_entity = [e01 + sep_token + e02 for e01, e02 in zip(subject_entities, object_entities)]
            tokens = self.tokenizer(
                concat_entity,
                sentences,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )

        return tokens

    def syllable_tokenizer(self, entities, sentences, max_seq_length):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []

        sep_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])
        pad_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])
        cls_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["cls_token"])

        if "xlm" in self.model_name:
            prefix = "▁"
        elif "bert" in self.model_name or "electra" in self.model_name:
            prefix = "##"

        for entity, sentence in zip(entities, sentences):
            now_index = 0
            input_ids = [pad_token_ids] * (max_seq_length - 1)
            attention_mask = [0] * (max_seq_length - 1)
            token_type_ids = [0] * (max_seq_length - 1)

            for e in entity:
                pre_syllable = "_"
                e = e.replace(" ", "_")
                for syllable in e:
                    if syllable == "_":
                        pre_syllable = "_"
                    if pre_syllable != "_":
                        if syllable not in [",", "."]:
                            syllable = prefix + syllable  # 중간 음절에는 모두 prefix를 붙입니다. (',', '.'에 대해서는 prefix를 붙이지 않습니다.)
                        # 이순신은 조선 -> [이, ##순, ##신, ##은, 조, ##선]
                    pre_syllable = syllable

                    input_ids[now_index] = self.tokenizer.convert_tokens_to_ids(syllable)
                    attention_mask[now_index] = 1
                    now_index += 1

                input_ids[now_index] = sep_token_ids
                attention_mask[now_index] = 1
                now_index += 1

            sentence = sentence[: max_seq_length - 2 - now_index].replace(" ", "_")
            pre_syllable = "_"
            for syllable in sentence:
                if syllable == "_":
                    pre_syllable = syllable
                if pre_syllable != "_":
                    if syllable not in [",", "."]:
                        syllable = prefix + syllable  # 중간 음절에는 모두 prefix를 붙입니다. (',', '.'에 대해서는 prefix를 붙이지 않습니다.)
                    # 이순신은 조선 -> [이, ##순, ##신, ##은, 조, ##선]
                pre_syllable = syllable

                input_ids[now_index] = self.tokenizer.convert_tokens_to_ids(syllable)
                attention_mask[now_index] = 1
                if "xlm" not in self.model_name or "roberta" not in self.model_name:
                    token_type_ids[now_index] = 1
                now_index += 1

            input_ids = [cls_token_ids] + input_ids
            input_ids[now_index + 1] = sep_token_ids
            token_type_ids = [0] + token_type_ids
            if "xlm" not in self.model_name or "roberta" not in self.model_name:
                token_type_ids[now_index + 1] = 1
            attention_mask = [1] + attention_mask
            attention_mask[now_index + 1] = 1

            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)

        return {
            "input_ids": torch.tensor(input_ids_list),
            "token_type_ids": torch.tensor(token_type_ids_list),
            "attention_mask": torch.tensor(attention_mask_list),
        }

    def preprocess(self, df):
        from utils.utils import label_to_num

        """
        기존 subject_entity, object entity string에서 word만 추출
            e.g. "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}" => 비틀즈
        train/dev set의 경우 label을 str ->  int
        """
        extract_entity = lambda row: eval(row)["word"].replace("'", "")
        df["subject_entity"] = df["subject_entity"].apply(extract_entity)
        df["object_entity"] = df["object_entity"].apply(extract_entity)

        if isinstance(df["label"].iloc[0], str):
            num_labels = label_to_num(df["label"].values)
            df["label"] = num_labels

        return df

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

            if self.train_ratio == 1.0:
                val_ratio = 0.2
                train_data, val_data = train_test_split(total_data, test_size=val_ratio)
                train_data = total_data
            else:
                train_data, val_data = train_test_split(total_data, train_size=self.train_ratio)

            # new dataframe
            train_df = self.preprocess(train_data)
            val_df = self.preprocess(val_data)

            self.train_dataset = CustomDataset(train_df)
            self.val_dataset = CustomDataset(val_df)
        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_df = self.preprocess(test_data)
            predict_df = self.preprocess(predict_data)

            self.test_dataset = CustomDataset(test_df)
            self.predict_dataset = CustomDataset(predict_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            collate_fn=self.batchify,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batchify,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batchify,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batchify,
            num_workers=self.num_workers,
        )

    @property
    def new_vocab_size(self):
        return self.new_token_count + self.tokenizer.vocab_size


class BaseKFoldDataModule(pl.LightningDataModule, ABC):
    """Essential for KFoldDataloader"""

    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


class KfoldDataloader(BaseKFoldDataModule, BaseDataloader):
    def __init__(self, config):
        super().__init__(config)

        self.shuffle = config.dataloader.shuffle
        self.num_folds = config.k_fold.num_folds
        self.train_fold = None
        self.val_fold = None
        self.fold_index = None

    def setup(self, stage="fit"):
        if stage == "fit":
            train_data = pd.read_csv(self.train_path)
            train_df = self.preprocess(train_data)
            self.train_dataset = CustomDataset(train_df)

        test_data = pd.read_csv(self.test_path)
        test_df = self.preprocess(test_data)
        self.test_dataset = CustomDataset(test_df)

        if stage == "predict":
            predict_data = pd.read_csv(self.predict_path)
            predict_df = self.preprocess(predict_data)
            self.predict_dataset = CustomDataset(predict_df)

    def setup_folds(self, num_folds) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index) -> None:
        self.fold_index = fold_index
        train_indices, val_indices = self.splits[self.fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.batchify)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, batch_size=self.batch_size, collate_fn=self.batchify)

    def __post_init__(cls):
        super().__init__()


class StratifiedDataloader(BaseDataloader):
    def __init__(self, config):
        super().__init__(config)
        assert self.train_ratio > 0.0 and self.train_ratio < 1.0

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

            split = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio)
            for train_idx, val_idx in split.split(total_data, total_data["label"]):
                train_data = total_data.loc[train_idx]
                val_data = total_data.loc[val_idx]

            # new dataframe
            train_df = self.preprocess(train_data)
            val_df = self.preprocess(val_data)

            self.train_dataset = CustomDataset(train_df)
            self.val_dataset = CustomDataset(val_df)
        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_df = self.preprocess(test_data)
            predict_df = self.preprocess(predict_data)

            self.test_dataset = CustomDataset(test_df)
            self.predict_dataset = CustomDataset(predict_df)
