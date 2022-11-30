import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from dataloader.dataset import RE_Dataset, RE_Collator, load_test_data, num_to_label


def inference(args, config):
    trainer = pl.Trainer(gpus=1, max_epochs=config.train.max_epoch, log_every_n_steps=1, deterministic=True)
    dataloader, model = utils.new_instance(config)
    if args.mode in ["inference", "i"]:
        model, _, __ = utils.load_model(args, config, dataloader, model)

    if args.mode in ["all", "a"]:
        model.load_from_checkpoint(config.path.best_model_path)

def inference(model, tokenizer, sentences, device):
    dataloader = DataLoader(
        sentences, batch_size=16, shuffle=False, collate_fn=RE_Collator(tokenizer)
    )
    model.eval()
    output_pred = []
    output_prob = []

    for data in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                token_type_ids=data["token_type_ids"].to(device),
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Load Tokenizer ###
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    ### Load Model ###
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.best_model_path
    )
    model.to(device)

    ### Load Dataset ###
    test_id, test_data, test_label = load_test_data(config.path.test_path)
    RE_test_dataset = RE_Dataset(test_data, test_label)

    ### Predict ###
    pred_answer, output_prob = inference(model, tokenizer, RE_test_dataset, device)
    pred_answer = num_to_label(pred_answer)

    output = pd.DataFrame(
        {
            "id": range(len(pred_answer)),
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    if not os.path.isdir("prediction"):
        os.mkdir("prediction")
    path = args.saved_model if args.saved_model is not None else config.path.best_model_path
    run_name = config.model.name + path.split("/")[-1]
    run_name = run_name.replace("/", "-")
    output.to_csv(f"./prediction/submission_{run_name}.csv", index=False)
