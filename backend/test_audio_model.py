import torch

from models.audio_cnn_lstm import AudioCNNLSTM


model = AudioCNNLSTM()
model.eval()

# Batch 1, channel 1, 128 Mel frequencies, 400 time steps
sample_audio = torch.randn(1, 1, 128, 400)

with torch.no_grad():
    output = model(sample_audio)

probabilities = torch.softmax(output, dim=1)

print("Input shape:", sample_audio.shape)
print("Output shape:", output.shape)
print("Fake probability:", probabilities[0][0].item())
print("Real probability:", probabilities[0][1].item())