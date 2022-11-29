import os
import pickle
import re
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer

class CustomDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        sentence = self.df['sentence'].iloc[idx]
        subject_entity = self.df['subject_entity'].iloc[idx]
        object_entity = self.df['object_entity'].iloc[idx]
        label = self.df['label'].iloc[idx]
        return sentence, subject_entity, object_entity, label

    def __len__(self):
        return len(self.df)

# ⭐⭐⭐MultipleHead⭐⭐⭐
class MultipleHeadDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        sentence = self.df['sentence'].iloc[idx]
        subject_entity = self.df['subject_entity'].iloc[idx]
        object_entity = self.df['object_entity'].iloc[idx]
        label = self.df['label'].iloc[idx]
        is_relation_label = self.df['is_relation_label'].iloc[idx]
        return sentence, subject_entity, object_entity, label, is_relation_label

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

        # self.tokenizer.model_max_length = 256
        self.new_token_count = 0
        if self.new_tokens != []:
            self.new_token_count += self.tokenizer.add_tokens(self.new_tokens, special_tokens=False)
        if self.new_special_tokens != []:
            self.new_token_count += self.tokenizer.add_tokens(self.new_special_tokens, special_tokens=True) 
    
    def batchify(self, batch):
        ''' data collator '''
        sentences, subject_entities, object_entities, labels = zip(*batch)

        outs = self.tokenize(sentences, subject_entities, object_entities)
        labels = torch.tensor(labels)
        return outs, labels

    def tokenize(self, sentences, subject_entities, object_entities):
        """
        tokenizer로 과제에 따라 tokenize 
        """
        sep_token = self.tokenizer.special_tokens_map["sep_token"]

        concat_entity = [e01 + sep_token + e02 for e01, e02 in zip(subject_entities, object_entities)]

        tokens = self.tokenizer(
            concat_entity,
            sentences,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )

        return tokens

    def preprocess(self, df):
        from utils.utils import label_to_num
        """
        기존 subject_entity, object entity string에서 word만 추출
            e.g. "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}" => 비틀즈
        train/dev set의 경우 label을 str ->  int
        """
        extract_entity = lambda row: eval(row)['word'].replace("'", "")
        df['subject_entity'] = df['subject_entity'].apply(extract_entity)
        df['object_entity'] = df['object_entity'].apply(extract_entity)

        if isinstance(df['label'].iloc[0], str): 
            num_labels = label_to_num(df['label'].values)
            df['label'] = num_labels
        
        return df

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

            if self.train_ratio == 1.0 :
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
    def __init__(self, k, config):
        super().__init__(config)
        self.shuffle = config.dataloader.shuffle
        self.num_splits = config.k_fold.num_splits
        self.k = k

    def setup(self, stage="fit"):
        if stage == "fit":
            self.total_data = pd.read_csv(self.train_path)
            kf = KFold(
                    n_splits=self.num_splits,
                    shuffle=self.shuffle,
            )
            self.train_sets, self.val_sets = [], []
            for train_index, val_index in kf.split(self.total_data):
                train_inputs, train_targets = self.preprocess(self.total_data.loc[train_index])
                val_inputs, val_targets = self.preprocess(self.total_data.loc[val_index])

                self.train_dataset = CustomDataset(train_inputs, train_targets)
                self.val_dataset = CustomDataset(val_inputs, val_targets)

                self.train_sets.append(self.train_dataset)
                self.val_sets.append(self.val_dataset)

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets = self.preprocess(test_data)
            predict_inputs, predict_targets = self.preprocess(predict_data)

            self.test_dataset = CustomDataset(test_inputs, test_targets)
            self.predict_dataset = CustomDataset(predict_inputs, predict_targets)
    
class StratifiedDataloader(BaseDataloader):
    def __init__(self, config):
        super().__init__(config)
        assert self.train_ratio > 0.0 and self.train_ratio <1.0

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

# ⭐⭐⭐MultipleHead⭐⭐⭐
class MultipleHeadDataloader(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.model_name = config.model.name
        self.batch_size = config.train.batch_size
        self.train_ratio = config.dataloader.train_ratio
        self.shuffle = config.dataloader.shuffle

        num_cpus = os.cpu_count()
        self.num_workers = num_cpus if self.batch_size//num_cpus <= num_cpus else int((self.batch_size//num_cpus) ** 0.5)

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

        self.new_token_count = 0
        if self.new_tokens != []:
            self.new_token_count += self.tokenizer.add_tokens(self.new_tokens)
            print(f"{self.new_token_count} new token(s) are added to the vocabulary.")
    
    def batchify(self, batch):
        """data collator"""
        sentences, subject_entities, object_entities, labels, is_relation_label= zip(*batch)
        
        outs = self.tokenize(sentences, subject_entities, object_entities,is_relation_label)
        labels = torch.tensor(labels)
        return outs, labels

    def tokenize(self, sentences, subject_entities, object_entities,is_relation_label):
        """
        tokenizer로 과제에 따라 tokenize 
        """
        sep_token = self.tokenizer.special_tokens_map["sep_token"]
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

    def preprocess(self, df):
        from utils.utils import label_to_num
        """
        기존 subject_entity, object entity string에서 word만 추출
            e.g. "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}" => 비틀즈
        train/dev set의 경우 label을 str ->  int
        """
        extract_entity = lambda row: eval(row)['word'].replace("'", "")
        df['subject_entity'] = df['subject_entity'].apply(extract_entity)
        df['object_entity'] = df['object_entity'].apply(extract_entity)
        
        # ⭐⭐⭐
        is_relation_label = [] 
        for label in df["label"]:
            if label != "no_relation":
                is_relation_label.append(1) # 1 → yes_relation
            else:
                is_relation_label.append(0) # 0 → no_relation
        # ⭐⭐⭐
        df['is_relation_label'] = is_relation_label

        if isinstance(df['label'].iloc[0], str): 
            num_labels = label_to_num(df['label'].values)
            df['label'] = num_labels

        return df 

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

            if self.train_ratio == 1.0 :
                val_ratio = 0.2
                train_data, val_data = train_test_split(total_data, test_size=val_ratio)
                train_data = total_data  
            else:
                train_data, val_data = train_test_split(total_data, train_size=self.train_ratio)

            # new dataframe 
            train_df = self.preprocess(train_data)
            val_df = self.preprocess(val_data)

            self.train_dataset = MultipleHeadDataset(train_df)
            self.val_dataset = MultipleHeadDataset(val_df)

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_df = self.preprocess(test_data)
            predict_df = self.preprocess(predict_data)

            self.test_dataset = MultipleHeadDataset(test_df)
            self.predict_dataset = MultipleHeadDataset(predict_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=self.shuffle, batch_size=self.batch_size, collate_fn=self.batchify, num_workers=self.num_workers,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.batchify, num_workers=self.num_workers,)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.batchify, num_workers=self.num_workers,)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, collate_fn=self.batchify, num_workers=self.num_workers,)

    @property
    def new_vocab_size(self):
        return self.new_token_count + self.tokenizer.vocab_size
