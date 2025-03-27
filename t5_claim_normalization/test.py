import os
import torch
import pandas as pd
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

BASE_DIR = Path("C:/Users/Hafid/Desktop/ML Assignment/t5_claim_normalization").resolve()
DEV_PATH = BASE_DIR / "data" / "dev-eng.csv"

BASE_MODELS = {
    "T5-small (no prompt)": BASE_DIR / "base_models" / "model_small_no_prompt",
    "T5-base (no prompt)": BASE_DIR / "base_models" / "model_base_manual_prompting"
}

PEFT_MODELS = {
    "T5-small (prompt-tuned)": BASE_DIR / "prompt_tuned" / "t5_small_peft",
    "T5-base (prompt-tuned)": BASE_DIR / "prompt_tuned" / "t5_base_peft"
}

# ✅ Map PEFT model names to their base model folders
PEFT_BASES = {
    "T5-small (prompt-tuned)": BASE_MODELS["T5-small (no prompt)"],
    "T5-base (prompt-tuned)": BASE_MODELS["T5-base (no prompt)"]
}

# ✅ Load dev data (first 5 posts)
df = pd.read_csv(DEV_PATH)
df = df[:5]

def make_hf_path(path: Path):
    return str(path.resolve().as_posix())  # Convert to safe forward-slash format

def generate_predictions(model_path, is_peft=False, base_path=None, add_prompt=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = make_hf_path(model_path)
    if is_peft:
        base_path = make_hf_path(base_path)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(base_path if is_peft else model_path, local_files_only=True)

    # Load model
    if is_peft:
        base_model = T5ForConditionalGeneration.from_pretrained(base_path, local_files_only=True)
        model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    model = model.to(device)
    model.eval()

    predictions = []
    for post in df["post"]:
        input_text = f"Normalize this claim: {post}" if add_prompt else post
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        output_ids = model.generate(**inputs, max_new_tokens=64)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred)

    return predictions

# ✅ Run inference for all models
results = {"Original Post": df["post"]}

for name, path in BASE_MODELS.items():
    print(f"🔍 Running {name}...")
    try:
        preds = generate_predictions(path, is_peft=False, add_prompt=True)
        results[name] = preds
    except Exception as e:
        print(f"❌ {name} failed: {e}")

for name, path in PEFT_MODELS.items():
    base_path = PEFT_BASES[name]
    print(f"🧪 Running {name}...")
    try:
        preds = generate_predictions(path, is_peft=True, base_path=base_path, add_prompt=False)
        results[name] = preds
    except Exception as e:
        print(f"❌ {name} failed: {e}")

# ✅ Save results to CSV
results_df = pd.DataFrame(results)
output_path = BASE_DIR / "test_output_sample.csv"
results_df.to_csv(output_path, index=False)

print(f"\n✅ Done! Results saved to: {output_path}")
print(results_df)
