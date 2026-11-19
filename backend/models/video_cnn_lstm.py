import torch
from torch import nn
from torchvision import models


class VideoCNNLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_classes: int = 2,
    ):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Remove ResNet's final classification layer
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = video.shape

        video = video.view(
            batch_size * sequence_length,
            channels,
            height,
            width,
        )

        # CNN processes every frame
        frame_features = self.cnn(video)

        frame_features = frame_features.view(
            batch_size,
            sequence_length,
            512,
        )

        # LSTM processes the sequence of frame features
        lstm_output, _ = self.lstm(frame_features)

        last_frame_output = lstm_output[:, -1, :]

        return self.classifier(last_frame_output)