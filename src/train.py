
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,

)
from typing import Dict
from huggingface_hub import login
from dotenv import load_dotenv

from src.contrains import LABEL2ID, ID2LABEL, MODEL_NAME, MODEL_DIR
from src.utils import compute_metrics, read_file_dataset
from src.dataset import HateDataset

load_dotenv()
login(os.getenv("HF_TOKEN"))

NUM_EPOCHS = 3
MAX_LEN = 128
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5


def main():
    train_set = read_file_dataset("./data/train_BIO_Word.csv")
    dev_set = read_file_dataset("./data/dev_BIO_Word.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    train_dataset = HateDataset(train_set["tokens"], train_set["labels"], tokenizer, LABEL2ID)
    dev_dataset = HateDataset(dev_set["tokens"], dev_set["labels"], tokenizer, LABEL2ID)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        weight_decay=1e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_strategy="steps",
        report_to="none"  # tránh gửi log đến wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.train()
    trainer.save_model(os.path.join(MODEL_DIR, "final_model"))

if __name__ == "__main__":
    main()
