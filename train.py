import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import random

from model.fusion import MultimodalFakeNews


# ---------------- CONFIG ----------------
EPOCHS = 5
LR = 0.001
VOCAB_SIZE = 5000
NUM_CLASSES = 2


# ---------------- TOKENIZER ----------------
def simple_tokenizer(text):
    tokens = text.lower().split()
    return torch.tensor(
        [hash(token) % VOCAB_SIZE for token in tokens],
        dtype=torch.long
    )


# ---------------- DUMMY DATASET ----------------
class DummyFakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = [
            ("Government announces new education policy", "test.jpg", 1),
            ("Aliens landed on Earth yesterday", "test.jpg", 0),
            ("New AI chip released by tech company", "test.jpg", 1),
            ("Man claims to time travel using phone", "test.jpg", 0),
        ]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, image_path, label = self.samples[idx]

        text_tensor = simple_tokenizer(text)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        return text_tensor, image_tensor, torch.tensor(label)


# ---------------- TRAINING LOOP ----------------
def train():
    model = MultimodalFakeNews()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    dataset = DummyFakeNewsDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    for epoch in range(EPOCHS):
        total_loss = 0

        for text, image, label in dataloader:
            text = text.unsqueeze(0)
            image = image
            label = label

            optimizer.zero_grad()
            output = model(text, image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model_weights.pth")
    print("Model training completed and weights saved.")


if __name__ == "__main__":
    train()
