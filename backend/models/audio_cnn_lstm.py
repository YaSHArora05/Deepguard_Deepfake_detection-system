import torch
from torch import nn


class AudioCNNLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()

        # CNN extracts features from the Mel spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(
            input_size=32 * 32,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        features = self.cnn(audio)

        batch_size, channels, frequency, time = features.shape

        # Change into: batch, time, features
        features = features.permute(0, 3, 1, 2)

        features = features.reshape(
            batch_size,
            time,
            channels * frequency,
        )

        lstm_output, _ = self.lstm(features)

        last_output = lstm_output[:, -1, :]

        return self.classifier(last_output)