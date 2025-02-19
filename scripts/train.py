import os

os.environ["WANDB_DISABLED"] = "true"

from datasets import load_from_disk
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# **Load preprocessed dataset**
dataset = load_from_disk("./data/processed_imdb")

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Fine-tune the model
trainer.train()
