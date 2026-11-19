import torch

from models.image_cnn import ImageCNN
from preprocess import preprocess_image


image_path = "sample.jpg"

try:
    image_tensor = preprocess_image(image_path)

    model = ImageCNN(num_classes=2)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    print("Input shape:", image_tensor.shape)
    print("Output shape:", output.shape)
    print("Fake probability:", probabilities[0][0].item())
    print("Real probability:", probabilities[0][1].item())

except Exception as error:
    print("Error:", error)