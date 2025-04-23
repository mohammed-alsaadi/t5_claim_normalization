import os
import torch
import argparse
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

def build_model_options(base_dir: Path):
    return {
        "1": {
            "name": "T5-small (no prompt)",
            "path": base_dir / "base_models" / "model_small_no_prompt",
            "is_peft": False,
            "add_prompt": True
        },
        "2": {
            "name": "T5-base (manual prompt)",
            "path": base_dir / "base_models" / "model_base_manual_prompting",
            "is_peft": False,
            "add_prompt": True
        },
        "3": {
            "name": "T5-small (prompt-tuned)",
            "path": base_dir / "prompt_tuned" / "t5_small_peft",
            "is_peft": True,
            "base": base_dir / "base_models" / "model_small_no_prompt",
            "add_prompt": False
        },
        "4": {
            "name": "T5-base (prompt-tuned)",
            "path": base_dir / "prompt_tuned" / "t5_base_peft",
            "is_peft": True,
            "base": base_dir / "base_models" / "model_base_manual_prompting",
            "add_prompt": False
        },
        "5": {
            "name": "T5-base (LoRA)",
            "path": base_dir / "fine_tuned" / "t5_base_lora",
            "is_peft": True,
            "base": base_dir / "base_models" / "model_base_manual_prompting",
            "add_prompt": False
        }
    }

def select_model(model_options):
    print("🔍 Select a model to use:")
    for key, value in model_options.items():
        print(f"  {key}. {value['name']}")
    choice = input("Enter model number (1-5): ").strip()

    if choice not in model_options:
        print("❌ Invalid choice. Exiting.")
        exit()

    config = model_options[choice]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default=r"C:\Users\hamod\github repo\t5_claim_normalization",
        help="Base directory containing model folders"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    model_options = build_model_options(base_dir)

    model, tokenizer, add_prompt = select_model(model_options)

    print("\n🧪 Model test ready. Type 'exit' to quit.")
    while True:
        post = input("\nEnter a social media post: ")
        if post.lower() == "exit":
            print("👋 Exiting...")
            break
        print("📝 Normalized Claim:", generate_claim(post, model, tokenizer, add_prompt))

if __name__ == "__main__":
    main()
