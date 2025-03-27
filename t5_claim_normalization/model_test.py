import os
import torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

# ✅ Define base path to all models
BASE_DIR = Path("C:/Users/Hafid/Desktop/ML Assignment/t5_claim_normalization").resolve()

# ✅ Define available models
MODEL_OPTIONS = {
    "1": {
        "name": "T5-small (no prompt)",
        "path": BASE_DIR / "base_models" / "model_small_no_prompt",
        "is_peft": False,
        "add_prompt": True
    },
    "2": {
        "name": "T5-base (no prompt)",
        "path": BASE_DIR / "base_models" / "model_base_manual_prompting",
        "is_peft": False,
        "add_prompt": True
    },
    "3": {
        "name": "T5-small (prompt-tuned)",
        "path": BASE_DIR / "prompt_tuned" / "t5_small_peft",
        "is_peft": True,
        "base": BASE_DIR / "base_models" / "model_small_no_prompt",
        "add_prompt": False
    },
    "4": {
        "name": "T5-base (prompt-tuned)",
        "path": BASE_DIR / "prompt_tuned" / "t5_base_peft",
        "is_peft": True,
        "base": BASE_DIR / "base_models" / "model_base_manual_prompting",
        "add_prompt": False
    }
}

def select_model():
    print("🔍 Select a model to use:")
    for key, value in MODEL_OPTIONS.items():
        print(f"  {key}. {value['name']}")
    choice = input("Enter model number (1-4): ").strip()
    
    if choice not in MODEL_OPTIONS:
        print("❌ Invalid choice. Exiting.")
        exit()

    config = MODEL_OPTIONS[choice]
    model_path = str(config["path"].resolve().as_posix())

    print(f"\n✅ Loading: {config['name']}")
    print(f"📂 Model path: {model_path}")

    if config["is_peft"]:
        base_path = str(config["base"].resolve().as_posix())
        tokenizer = T5Tokenizer.from_pretrained(base_path, local_files_only=True)
        base_model = T5ForConditionalGeneration.from_pretrained(base_path, local_files_only=True)
        model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer, config["add_prompt"]

def generate_claim(post: str, model, tokenizer, add_prompt: bool) -> str:
    input_text = f"Normalize this claim: {post}" if add_prompt else post
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    output = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    model, tokenizer, add_prompt = select_model()

    print("\n🧪 Model test ready. Type 'exit' to quit.")
    while True:
        post = input("\nEnter a social media post: ")
        if post.lower() == "exit":
            print("👋 Exiting...")
            break
        print("📝 Normalized Claim:", generate_claim(post, model, tokenizer, add_prompt))

if __name__ == "__main__":
    main()
