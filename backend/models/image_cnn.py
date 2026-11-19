import torch
from torch import nn
from torchvision import models


class ImageCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Pretrained CNN that extracts image features
        self.cnn = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Replace original classifier with Real/Fake classifier
        input_features = self.cnn.fc.in_features

        self.cnn.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(input_features, num_classes)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.cnn(image)