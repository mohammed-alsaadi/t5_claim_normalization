import argparse
import os
import torch
import gc
import pandas as pd
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from t5_claim_normalization import TRAIN_CSV_PATH, ensure_datasets
import requests

def download_if_url(path_or_url, dest_path):
    if path_or_url is None:
        raise ValueError("Path or URL is None.")
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        print(f"Downloading dataset from {path_or_url}...")
        response = requests.get(path_or_url)
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {dest_path}")
        return dest_path
    return path_or_url

def preprocess_function(examples):
    prompt_text = "Normalize this claim: "

    inputs = [prompt_text + post for post in examples["post"]]
    targets = examples["normalized claim"]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(train_path=None, dev_path=None):
    # ensure_datasets()

    torch.cuda.empty_cache()
    gc.collect()

    global max_input_length, max_target_length
    max_input_length = 32
    max_target_length = 32

    train_file = download_if_url(train_path, "data/train.csv")
    dev_file = download_if_url(dev_path, "data/dev.csv")

    # train_csv_path = "data/train-eng.csv"
    # dev_csv_path = "data/dev-eng.csv"

    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    tokenized_train = train_dataset.map(preprocess_function, batched=True, num_proc=4)
    tokenized_dev = dev_dataset.map(preprocess_function, batched=True, num_proc=4)

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to("cuda")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        logging_steps=50,
        fp16=True,
        report_to="none",
        optim="adamw_torch_fused",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    torch.cuda.empty_cache()
    gc.collect()

    print("Starting training...")
    trainer.train()

    print("Evaluating on development set...")
    results = trainer.evaluate()
    print("Evaluation Results:", results)

    output_dir = "/model_result/t5_claim_normalization"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/raw/main/task2/data/train/train-eng.csv",
        help="Path or URL to training data",
    )
    parser.add_argument(
        "--dev_path",
        default="https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/raw/main/task2/data/dev/dev-eng.csv",
        help="Path or URL to dev data",
    )
    args = parser.parse_args()
    main(args.train_path, args.dev_path)

if __name__ == "__main__":
    cli()

