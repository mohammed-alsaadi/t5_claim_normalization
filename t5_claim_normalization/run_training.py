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


def preprocess_function(examples):
    inputs = examples["post"]
    targets = examples["normalized claim"]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # Ensure dataset is available
    ensure_datasets()

    # Free up GPU memory before loading the model (To avoid OOM errors in the Google Collab environment)
    torch.cuda.empty_cache()
    gc.collect()

    # Here we reduce sequence lengths for minimal memory usage
    global max_input_length, max_target_length
    max_input_length = 32
    max_target_length = 32

    # Load dataset
    train_csv_path = "data/train-eng.csv"
    dev_csv_path = "data/dev-eng.csv"

    train_df = pd.read_csv(train_csv_path)
    dev_df = pd.read_csv(dev_csv_path)

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    # Load tokenizer
    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Tokenize datasets
    tokenized_train = train_dataset.map(preprocess_function, batched=True, num_proc=4)
    tokenized_dev = dev_dataset.map(preprocess_function, batched=True, num_proc=4)

    # Load model with memory-optimized settings
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.gradient_checkpointing_enable()  # This allows us to save memory even further
    model.config.use_cache = False
    model.to("cuda")

    # Trainer setup with minimal memory usage
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

    # Start training
    print("Starting training...")
    trainer.train()

    # Evaluate model
    print("Evaluating on development set...")
    results = trainer.evaluate()
    print("Evaluation Results:", results)

    # Save final model
    output_dir = "/model_result/t5_claim_normalization"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
