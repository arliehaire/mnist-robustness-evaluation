import torch
import torch.nn as nn


class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (B, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # (B, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # (B, 64, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # (B, 64*7*7)
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
