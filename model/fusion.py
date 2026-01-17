import torch
import torch.nn as nn

from model.transformer import TransformerEncoder
from model.cnn import SimpleCNN


class MultimodalFakeNews(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = TransformerEncoder()
        self.image_model = SimpleCNN()

        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, text, image):
        text_feat = self.text_model(text)
        image_feat = self.image_model(image)

        fused = torch.cat([text_feat, image_feat], dim=1)
        return self.classifier(fused)


# 🔥 THIS LINE WAS MISSING OR WRONG
model = MultimodalFakeNews()
model.eval()

