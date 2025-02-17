import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def get_model_dir():
    candidate = os.path.join(os.getcwd(), "model")
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    else:
        raise FileNotFoundError("Model folder not found in the current working directory. Please run the script from the project root.")

MODEL_DIR = get_model_dir()

print(f"Loading model from: {MODEL_DIR}")

# Load the tokenizer and model from the model directory
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def generate_claim(post: str) -> str:
    inputs = tokenizer(post, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    output = model.generate(**inputs, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    print("Model test started. Type 'exit' to quit.")
    while True:
        post = input("\nEnter a social media post: ")
        if post.lower() == "exit":
            print("Exiting...")
            break
        print("Normalized Claim:", generate_claim(post))

if __name__ == "__main__":
    main()
