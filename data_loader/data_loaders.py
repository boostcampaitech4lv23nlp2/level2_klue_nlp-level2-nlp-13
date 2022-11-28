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
    def __init__(self, pair_dataset, labels, is_relation_labels):
        self.pair_dataset = pair_dataset
        self.labels = labels
        self.is_relation_labels = is_relation_labels

    def __getitem__(self, idx):
        if len(self.labels) == 0:
            item = [val[idx].clone().detach() for _, val in self.pair_dataset.items()]
        else:
            item = [val[idx].clone().detach() for _, val in self.pair_dataset.items()]
            item.append(self.labels[idx])
            item.append(self.is_relation_labels[idx])
        return item

    def __len__(self):
        return len(self.pair_dataset["input_ids"])
        
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
        is_relation_label = [] # ⭐

        for sub, obj in zip(df["subject_entity"], df["object_entity"]):
            sub = eval(sub)
            obj = eval(obj)

            subject_entity.append(sub["word"].replace("'", ""))
            object_entity.append(obj["word"].replace("'", ""))

        # ⭐⭐⭐
        for label in df["label"]:
            if label != "no_relation":
                is_relation_label.append(1) # 1 → yes_relation
            else:
                is_relation_label.append(0) # 0 → no_relation

        preprocessed_df = pd.DataFrame(
            {
                "sentence": df["sentence"],
                "subject_entity": subject_entity,
                "object_entity": object_entity,
                "is_relation_label": is_relation_label,
                "label": df["label"],
            }
        )

        try:
            if preprocessed_df["label"].iloc[0] == 100:  # test_data인 경우
                targets = []
                is_relation_targets = []
            else:
                targets = preprocessed_df["label"].values.tolist()
                targets = utils.label_to_num(targets)
                is_relation_targets = preprocessed_df["is_relation_label"].values.tolist() # ⭐
        except:
            targets = []
            is_relation_targets = []

        inputs = self.tokenizing(preprocessed_df)

        return inputs, targets, is_relation_targets # ⭐

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

            split = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=42)
            for train_idx, val_idx in split.split(total_data, total_data["label"]):
                train_data = total_data.loc[train_idx]
                val_data = total_data.loc[val_idx]

            train_inputs, train_targets, train_is_relation_targets = self.preprocessing(train_data)
            val_inputs, val_targets, val_is_relation_targets = self.preprocessing(val_data)
            print("train data len : ", len(train_inputs["input_ids"]))
            print("valid data len : ", len(val_inputs["input_ids"]))

            self.train_dataset = MultipleHeadDataset(train_inputs, train_targets, train_is_relation_targets)
            self.val_dataset = MultipleHeadDataset(val_inputs, val_targets, val_is_relation_targets)

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets, test_is_relation_targets = self.preprocessing(test_data) # ⭐
            predict_inputs, predict_targets, predict_is_relation_targets = self.preprocessing(predict_data) # ⭐
            print("predict data len : ", len(predict_inputs["input_ids"]))

            self.test_dataset = MultipleHeadDataset(test_inputs, test_targets, test_is_relation_targets) # ⭐
            self.predict_dataset = MultipleHeadDataset(predict_inputs, predict_targets, predict_is_relation_targets) # ⭐

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