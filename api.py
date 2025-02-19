from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize FastAPI
app = FastAPI()

# Load model & tokenizer
model_path = "./models/deberta_sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.post("/predict/")
async def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return {"positive": prediction[1], "negative": prediction[0]}
