import os
import pickle
import re
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from tqdm.auto import tqdm
from ..utils import utils


class CustomDataset(Dataset):

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
        # self.use_add_token = use_add_token
        # if self.use_add_token:
        #     self.add_token = [
        #         "<PERSON>",
        #     ]
        #     self.new_token_count = self.tokenizer.add_tokens(self.add_token)
        # else:
        #     self.new_token_count = 0
        # self.swap = use_swap

        # self.target_columns = ["label"]
        # self.delete_columns = ["id"]
        # self.text_columns = ["sentence", "subject_entity", "object_entity"]
    
    def collate(self, batch):
        sentences, subject_entities, object_entities, labels = zip(*batch)

        tokens_dict = self.tokenize(sentences, subject_entities, object_entities,)

        item = {key: val.clone().detach() for key, val in tokens_dict.items()}
        item['labels'] = torch.tensor(labels)
        return item

    def tokenize(self, sentences, subject_entities, object_entities,):
        """
        tokenizer로 과제에 따라 tokenize 
        """
        sep_token = self.tokenizer.special_tokens_map["sep_token"]

        concat_entity = [e01 + sep_token + e02 for e01, e02 in zip(subject_entities, object_entities)]

        # if self.use_add_token:
        #     text = utils.text_preprocessing(text)

        tokens = self.tokenizer(
            concat_entity,
            sentences,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return tokens

    def preprocess(self, df):
        """
        기존 subject_entity, object entity string에서 word만 추출
        e.g. "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}" => 비틀즈
        """
        # df = df.drop(columns=self.delete_columns)
        extract_entity = lambda row: eval(row)['word'].replace("'", "")
        df['subject_entity'] = df['subject_entity'].apply(extract_entity)
        df['object_entity'] = df['object_entity'].apply(extract_entity)

        # subject_entity = []
        # object_entity = []

        # for sub, obj in zip(df["subject_entity"], df["object_entity"]):
        #     sub = eval(sub)
        #     obj = eval(obj)

        #     subject_entity.append(sub["word"].replace("'", ""))
        #     object_entity.append(obj["word"].replace("'", ""))

        # preprocessed_df = pd.DataFrame(
        #     {
        #         "sentence": df["sentence"],
        #         "subject_entity": subject_entity,
        #         "object_entity": object_entity,
        #         "label": df["label"],
        #     }
        # )

        try:
            if df["label"].iloc[0] == 100:  # test_data인 경우
                targets = []
            else:
                targets = df["label"].values.tolist()
                targets = utils.label_to_num(targets)
        except:
            targets = []

        inputs = df['sentence'].values.tolist()

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

            split = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=1004)
            for train_idx, val_idx in split.split(total_data, total_data["label"]):
                train_data = total_data.loc[train_idx]
                val_data = total_data.loc[val_idx]

            train_inputs, train_targets = self.preprocess(train_data)
            val_inputs, val_targets = self.preprocess(val_data)
            print("train data len : ", len(train_inputs["input_ids"]))
            print("valid data len : ", len(val_inputs["input_ids"]))

            self.train_dataset = CustomDataset(train_inputs, train_targets)
            self.val_dataset = CustomDataset(val_inputs, val_targets)

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets = self.preprocess(test_data)
            predict_inputs, predict_targets = self.preprocess(predict_data)
            print("predict data len : ", len(predict_inputs["input_ids"]))

            self.test_dataset = CustomDataset(test_inputs, test_targets)
            self.predict_dataset = CustomDataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, collate_fn=self.collate)
    
    @property
    def new_vocab_size(self):
        return self.new_token_count + self.tokenizer.vocab_size


class KfoldDataloader(Dataloader):
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

        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            train_ratio=1.0,
            shuffle=shuffle,
            train_path=train_path,
            test_path=test_path,
            predict_path=predict_path,
            use_swap=use_swap,
            use_add_token=False,
        )
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
        # self.use_preprocessing = use_preprocessing
        # if self.use_preprocessing:
        #     self.add_token = [
        #         "<PERSON>",
        #         "...",
        #         # "!!!",
        #         # "???",
        #         "ㅎㅎㅎ",
        #         "ㅋㅋㅋ",
        #         "ㄷㄷㄷ",
        #     ]
        # else:
        #     self.add_token = [
        #         "<PERSON>",
        #     ]

        self.new_token_count = self.tokenizer.add_tokens(self.add_token)
        # self.swap = use_swap

    def prepare_data(self):
        total_data = pd.read_csv(self.train_path)
        kf = KFold(
                n_splits=self.num_splits,
                shuffle=self.shuffle,
                random_state=self.split_seed,
        )
        self.split_indices = [s for s in kf.split(total_data)]

    def setup(self, stage="fit"):
        if stage == "fit":
            train_indexes, val_indexes = self.split_indices[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # print("Number of splits: \n", self.num_splits)

            train_inputs, train_targets = self.preprocess(total_data.loc[train_indexes])
            valid_inputs, valid_targets = self.preprocess(total_data.loc[val_indexes])

            train_dataset = CustomDataset(train_inputs, train_targets)
            valid_dataset = CustomDataset(valid_inputs, valid_targets)

            self.train_dataset = train_dataset
            self.val_dataset = valid_dataset

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets = self.preprocess(test_data)
            predict_inputs, predict_targets = self.preprocess(predict_data)

            self.test_dataset = CustomDataset(test_inputs, test_targets)
            self.predict_dataset = CustomDataset(predict_inputs, predict_targets)

