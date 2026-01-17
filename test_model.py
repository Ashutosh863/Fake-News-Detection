import torch

from model.fusion import model
from utils.image_utils import preprocess_image


# ---------------- TOKENIZER ----------------
def simple_tokenizer(text, vocab_size=5000):
    tokens = text.lower().split()
    return torch.tensor(
        [[hash(token) % vocab_size for token in tokens]],
        dtype=torch.long
    )


# ---------------- INPUTS ----------------
text = "Breaking news: scientists discover water on Mars"
image_path = "test.jpg"  # must exist in project root

text_tensor = simple_tokenizer(text)
image_tensor = preprocess_image(image_path)


# ---------------- PREDICTION ----------------
with torch.no_grad():
    logits = model(text_tensor, image_tensor)
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs).item()


# ---------------- OUTPUT ----------------
print("LOGITS:", logits)
print("PROBABILITIES:", probs)
print("PREDICTION:", "REAL" if prediction == 1 else "FAKE")
