import os
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    )
from huggingface_hub import login
from dotenv import load_dotenv
from src.utils import read_file_dataset, compute_metrics
from src.dataset import HateDataset
from src.contrains import MODEL_NAME, LABEL2ID, ID2LABEL

load_dotenv()
print(os.getenv("HF_TOKEN"))
login(os.getenv("HF_TOKEN"))

# Load Model and Tokenizer
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, 
                                                        label2id=LABEL2ID, 
                                                        id2label=ID2LABEL)

tokenizer = AutoTokenizer.from_pretrained("./models/train_models/checkpoint-1112", use_fast=False)

# Load test dataset
test_set = read_file_dataset("./data/test_BIO_Word.csv")
test_dataset = HateDataset(test_set["tokens"], test_set["labels"], tokenizer, LABEL2ID)

train_args = TrainingArguments(
    output_dir="./eval_results",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=train_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
results = trainer.evaluate(test_dataset)

print("\nEvaluation Results:")
for key, value in results.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

with open("./eval_results.json", "w") as f:
    f.write(json.dumps(results, indent=4))