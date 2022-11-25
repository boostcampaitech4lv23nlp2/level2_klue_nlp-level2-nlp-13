import os
import pickle
import re

import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


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
        self.new_tokens = list(config.tokenizer.new_tokens)
        self.new_special_tokens = list(config.tokenizer.new_special_tokens)
        self.max_length = config.tokenizer.max_len
        self.use_syllable_tokenize = config.tokenizer.syllable

        self.train_path = config.path.train_path
        self.test_path = config.path.test_path
        self.predict_path = config.path.predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        assert isinstance(self.train_ratio, float) and self.train_ratio > 0.0 and self.train_ratio <= 1.0

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

        if self.model_name in model_list["bert"]:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        elif self.model_name in model_list["electra"]:
            self.tokenizer = transformers.ElectraTokenizer.from_pretrained(self.model_name)
        elif self.model_name in model_list["roberta"]:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
        elif self.model_name in model_list["funnel"]:
            self.tokenizer = transformers.FunnelTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        self.new_token_count = 0
        if self.new_tokens != []:
            self.new_token_count += self.tokenizer.add_tokens(self.new_tokens, special_tokens=False)
        if self.new_special_tokens != []:
            self.new_token_count += self.tokenizer.add_tokens(self.new_special_tokens, special_tokens=True)

    def batchify(self, batch):
        """data collator"""
        sentences, subject_entities, object_entities, labels = zip(*batch)

        outs = self.tokenize(
            sentences,
            subject_entities,
            object_entities,
        )
        input_ids = outs["input_ids"]
        token_type_ids = outs["token_type_ids"]
        attention_mask = outs["attention_mask"]
        labels = torch.tensor(labels)
        return input_ids, token_type_ids, attention_mask, labels

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
                            syllable = "##" + syllable  # 중간 음절에는 모두 prefix를 붙입니다. (',', '.'에 대해서는 prefix를 붙이지 않습니다.)
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
                        syllable = "##" + syllable  # 중간 음절에는 모두 prefix를 붙입니다. (',', '.'에 대해서는 prefix를 붙이지 않습니다.)
                    # 이순신은 조선 -> [이, ##순, ##신, ##은, 조, ##선]
                pre_syllable = syllable

                input_ids[now_index] = self.tokenizer.convert_tokens_to_ids(syllable)
                attention_mask[now_index] = 1
                token_type_ids[now_index] = 1
                now_index += 1

            input_ids = [cls_token_ids] + input_ids
            input_ids[now_index + 1] = sep_token_ids
            token_type_ids = [0] + token_type_ids
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

            if self.train_ratio < 1.0:
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.batchify)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.batchify)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.batchify)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, collate_fn=self.batchify)

    @property
    def new_vocab_size(self):
        return self.new_token_count + self.tokenizer.vocab_size


class KfoldDataloader(BaseDataloader):
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
        new_tokens=None,
        new_special_tokens=None,
    ):

        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            train_ratio=1.0,
            shuffle=shuffle,
            train_path=train_path,
            test_path=test_path,
            predict_path=predict_path,
            new_tokens=new_tokens,
            new_special_tokens=new_special_tokens,
        )
        self.k = k
        self.num_splits = num_splits

    def prepare_data(self):
        self.total_data = pd.read_csv(self.train_path)
        kf = KFold(
            n_splits=self.num_splits,
            shuffle=self.shuffle,
        )
        self.split_indices = [s for s in kf.split(self.total_data)]

    def setup(self, stage="fit"):
        if stage == "fit":
            train_indexes, val_indexes = self.split_indices[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            train_inputs, train_targets = self.preprocess(self.total_data.loc[train_indexes])
            valid_inputs, valid_targets = self.preprocess(self.total_data.loc[val_indexes])

            self.train_dataset = CustomDataset(train_inputs, train_targets)
            self.val_dataset = CustomDataset(valid_inputs, valid_targets)

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets = self.preprocess(test_data)
            predict_inputs, predict_targets = self.preprocess(predict_data)

            self.test_dataset = CustomDataset(test_inputs, test_targets)
            self.predict_dataset = CustomDataset(predict_inputs, predict_targets)


class StratifiedDataloader(BaseDataloader):
    def __init__(
        self,
        model_name,
        batch_size,
        train_ratio,
        shuffle,
        train_path,
        test_path,
        predict_path,
        new_tokens=None,
        new_special_tokens=None,
    ):
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            train_ratio=train_ratio,
            shuffle=shuffle,
            train_path=train_path,
            test_path=test_path,
            predict_path=predict_path,
            new_tokens=new_tokens,
            new_special_tokens=new_special_tokens,
        )

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
