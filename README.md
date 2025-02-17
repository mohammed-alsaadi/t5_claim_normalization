# T5 Claim Normalization

A T5-based claim normalization model for processing and normalizing social media posts.

## 🚀 Features
- Fine-tunes a **T5 model** on a dataset of social media posts.
- **Tokenizes** and **trains** using the `transformers` library.
- Evaluates performance using the **METEOR score**.
- Includes an **interactive testing script**.

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/t5_claim_normalization.git
cd t5_claim_normalization
```

### 2️⃣ Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install .
```

---

## 📜 Usage

### 🔹 Train the Model
```bash
train
```
This will:
- Load the dataset (`data/train-eng.csv` & `data/dev-eng.csv`).
- Train the model using `transformers.Trainer`.
- Save the final model to `model_result/t5_claim_normalization`.

### 🔹 Evaluate the Model
```bash
evaluate
```
This will:
- Load the trained model.
- Generate predictions on the **dev set**.
- Compute the **METEOR score**.

### 🔹 Test the Model (Interactive CLI)
```bash
test_model
```
You can enter a **social media post**, and the model will return a **normalized claim**.

---

## 📂 Project Structure
```
t5_claim_normalization/
│── data/                         # Directory for datasets
│   ├── train-eng.csv             # Training dataset
│   ├── dev-eng.csv               # Development dataset
│   ├── test-eng.csv              # Testing dataset
│
│── model/                        # Directory containing the trained model
│   ├── added_tokens.json
│   ├── config.json
│   ├── generation_config.json
│   ├── model_test.py             # Interactive testing script
│   ├── model.safetensors         # Trained model weights
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
│
│── evaluate_meteor.py            # Evaluation script
│── pyproject.toml                # Project metadata and dependencies
│── README.md                     # Documentation
│── requirements.txt              # List of required dependencies
│── run_training.py               # Model training script
```

---

## ⚙️ Dependencies
- `numpy`
- `pandas`
- `datasets`
- `transformers`
- `torch`
- `nltk`
- `sentencepiece`
- `accelerate`
- `protobuf`

Install manually with:
```bash
pip install numpy pandas datasets transformers torch nltk sentencepiece accelerate
```

---

## 👤 Author
- **Mohammed Alsaadi**
- Contact: [msa1045@usnh.edu]
