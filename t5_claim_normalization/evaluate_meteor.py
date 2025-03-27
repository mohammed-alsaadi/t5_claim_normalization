from nltk.translate.meteor_score import meteor_score
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

def compute_meteor(predictions, references):
    tokenized_predictions = [pred.split() for pred in predictions]
    tokenized_references = [[ref.split()] for ref in references]

    scores = [meteor_score(ref, pred) for pred, ref in zip(tokenized_predictions, tokenized_references)]
    return sum(scores) / len(scores)

def main():
    MODEL_DIR = os.path.abspath("./model")
    DATA_DIR = os.path.abspath("./data")
    print(f"Loading model from {MODEL_DIR}...")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

    dev_csv_path = os.path.join(DATA_DIR, "dev-eng.csv")
    print(f"Loading validation dataset from {dev_csv_path}...")
    dev_df = pd.read_csv(dev_csv_path)

    print("Generating predictions...")
    predictions = []
    for post in dev_df["post"]:
        inputs = tokenizer(post, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        output = model.generate(**inputs, max_length=50)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(prediction)

    print("Computing METEOR score...")
    meteor_result = compute_meteor(predictions, dev_df["normalized claim"])
    print(f"METEOR Score: {meteor_result:.4f}")

if __name__ == "__main__":
    main()
