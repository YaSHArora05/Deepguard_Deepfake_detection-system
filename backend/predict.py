from pathlib import Path

import torch

from models.image_cnn import ImageCNN
from models.video_cnn_lstm import VideoCNNLSTM
from models.audio_cnn_lstm import AudioCNNLSTM
from preprocess import preprocess_file


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_FOLDER = Path(__file__).resolve().parent / "weights"

MODEL_CONFIG = {
    "image": {
        "model": ImageCNN(),
        "weights": WEIGHTS_FOLDER / "image_model.pth",
    },
    "video": {
        "model": VideoCNNLSTM(),
        "weights": WEIGHTS_FOLDER / "video_model.pth",
    },
    "audio": {
        "model": AudioCNNLSTM(),
        "weights": WEIGHTS_FOLDER / "audio_model.pth",
    },
}

CLASS_NAMES = {
    0: "fake",
    1: "real",
}


def load_model(file_type: str):
    config = MODEL_CONFIG[file_type]
    model = config["model"].to(DEVICE)
    weights_path = config["weights"]

    if not weights_path.exists():
        return model, False

    checkpoint = torch.load(
        weights_path,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, True


def run_prediction(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> dict:
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )

    index = predicted_index.item()

    return {
        "prediction": CLASS_NAMES[index],
        "confidence": round(confidence.item(), 4),
        "fake_probability": round(
            probabilities[0][0].item(),
            4,
        ),
        "real_probability": round(
            probabilities[0][1].item(),
            4,
        ),
    }


def predict_file(file_path: str) -> dict:
    file_type, tensor = preprocess_file(file_path)

    model, is_trained = load_model(file_type)

    if not is_trained:
        return {
            "file_type": file_type,
            "prediction": "unavailable",
            "message": f"{file_type.capitalize()} model is not trained yet",
        }

    result = run_prediction(model, tensor)
    result["file_type"] = file_type

    return result