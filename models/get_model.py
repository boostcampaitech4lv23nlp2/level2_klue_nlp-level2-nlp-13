from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
)
from models.entity_embeddings import CustomRobertaEmbeddings

def get_model(config, tokenizer, added_token_num):
    if config.model.arch == "RobertaLarge":
        model_config = AutoConfig.from_pretrained("klue/roberta-large")
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.name, config=model_config
    )

    elif config.model.arch == "RobertaLargeWithTypedEntityTokens":
        model_config = AutoConfig.from_pretrained("klue/roberta-large")
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.name, config=model_config
        )
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)


    elif config.model.arch == "RobertaLargeEWithCustomEmbeddings":
        model_config = AutoConfig.from_pretrained("klue/roberta-large")
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(
            "klue/roberta-large", config=model_config
        )
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

        embedding_config = AutoConfig.from_pretrained("klue/roberta-large")
        model.roberta.embeddings = CustomRobertaEmbeddings(embedding_config)

    return model