import torch

from models.video_cnn_lstm import VideoCNNLSTM


model = VideoCNNLSTM(num_classes=2)
model.eval()

# Batch 1, 20 frames, RGB, 224 × 224
sample_video = torch.randn(1, 20, 3, 224, 224)

with torch.no_grad():
    output = model(sample_video)

probabilities = torch.softmax(output, dim=1)

print("Input shape:", sample_video.shape)
print("Output shape:", output.shape)
print("Fake probability:", probabilities[0][0].item())
print("Real probability:", probabilities[0][1].item())