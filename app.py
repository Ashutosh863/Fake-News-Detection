from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np

from model.fusion import model
from utils.image_utils import preprocess_image

app = FastAPI(title="Fake News Detection (From Scratch)")

def simple_tokenizer(text, vocab_size=5000):
    tokens = text.lower().split()
    return torch.tensor(
        [[hash(token) % vocab_size for token in tokens]],
        dtype=torch.long
    )

@app.post("/predict")
async def predict(text: str, image: UploadFile = File(...)):
    text_tensor = simple_tokenizer(text)
    image_tensor = preprocess_image(image.file)

    with torch.no_grad():
        output = model(text_tensor, image_tensor)
        probs = torch.softmax(output, dim=1)
        label = torch.argmax(probs).item()

    return {
        "prediction": "REAL" if label == 1 else "FAKE",
        "confidence": probs.tolist()
    }
