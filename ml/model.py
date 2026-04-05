import torch
import torch.nn as nn


class GreekLetterCNN(nn.Module):
    def __init__(self, num_classes, channels=[32, 64], hidden_size=128, dropout=0.3):
        super().__init__()

        layers = []
        in_channels = 1

        for ch in channels:
            layers += [
                nn.Conv2d(in_channels, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
            in_channels = ch

        self.features = nn.Sequential(*layers)

        # compute feature map size after pooling
        # input = 64x64, each pool halves size
        size = 64 // (2 ** len(channels))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * size * size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x