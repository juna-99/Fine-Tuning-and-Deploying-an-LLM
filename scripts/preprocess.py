from datasets import load_dataset
import re
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("imdb")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# Apply cleaning (batch processing)
dataset = dataset.map(
    lambda batch: {"text": [clean_text(text) for text in batch["text"]]}, batched=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)


# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


# Apply tokenization
dataset = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch (AFTER tokenization)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# **Save dataset**
dataset.save_to_disk("./data/processed_imdb")
