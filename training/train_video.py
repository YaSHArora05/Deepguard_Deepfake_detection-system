from pathlib import Path
import re
import sys

import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.video_cnn_lstm import VideoCNNLSTM

DATASET_FOLDER = PROJECT_ROOT / "dataset" / "video" / "1000_videos"
TRAIN_FOLDER = DATASET_FOLDER / "train"
VALIDATION_FOLDER = DATASET_FOLDER / "validation"

CACHE_FOLDER = PROJECT_ROOT / "dataset" / "video" / "frame_feature_cache"
WEIGHTS_FOLDER = PROJECT_ROOT / "backend" / "weights"
WEIGHTS_PATH = WEIGHTS_FOLDER / "video_model.pth"

IMAGE_SIZE = 112
SEQUENCE_LENGTH = 20
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0003
NUM_WORKERS = 0
FEATURE_BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAPPING = {"fake": 0, "real": 1}

frame_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def get_video_id(file_path: Path) -> str:
    stem = file_path.stem

    if "_" not in stem:
        raise ValueError(f"Invalid frame filename: {file_path.name}")

    return stem.rsplit("_", 1)[0]


def get_frame_number(file_path: Path) -> int:
    match = re.search(r"_(\d+)$", file_path.stem)

    if match is None:
        raise ValueError(f"Could not read frame number: {file_path.name}")

    return int(match.group(1))


def collect_video_groups(split_folder: Path):
    groups = []

    for class_name, label in CLASS_MAPPING.items():
        class_folder = split_folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(f"Missing folder: {class_folder}")

        grouped_frames = {}

        for file_path in class_folder.glob("*.png"):
            video_id = get_video_id(file_path)
            grouped_frames.setdefault(video_id, []).append(file_path)

        for video_id, frame_paths in grouped_frames.items():
            frame_paths.sort(key=get_frame_number)

            groups.append(
                {
                    "video_id": video_id,
                    "frames": frame_paths,
                    "label": label,
                    "class_name": class_name,
                }
            )

    return groups


def sample_frame_paths(frame_paths, sequence_length):
    if len(frame_paths) == 0:
        raise ValueError("Video group contains no frames")

    if len(frame_paths) >= sequence_length:
        indexes = torch.linspace(
            0,
            len(frame_paths) - 1,
            steps=sequence_length,
        ).long()

        return [frame_paths[index.item()] for index in indexes]

    selected_paths = list(frame_paths)

    while len(selected_paths) < sequence_length:
        selected_paths.append(selected_paths[-1])

    return selected_paths


def load_frame_tensor(frame_path: Path) -> torch.Tensor:
    image = Image.open(frame_path).convert("RGB")
    return frame_transform(image)


def create_feature_cache(model, split_folder, split_name):
    video_groups = collect_video_groups(split_folder)

    split_cache_folder = CACHE_FOLDER / split_name
    split_cache_folder.mkdir(parents=True, exist_ok=True)

    cache_files = []
    failed_groups = []

    model.eval()

    progress_bar = tqdm(
        video_groups,
        desc=f"Caching {split_name} sequences",
        unit="video",
    )

    for group in progress_bar:
        video_id = group["video_id"]
        class_name = group["class_name"]
        label = group["label"]

        class_cache_folder = split_cache_folder / class_name
        class_cache_folder.mkdir(parents=True, exist_ok=True)

        cache_path = class_cache_folder / f"{video_id}.pt"

        progress_bar.set_postfix(current=f"{class_name}/{video_id}")

        if cache_path.exists():
            cache_files.append(cache_path)
            continue

        try:
            selected_paths = sample_frame_paths(
                group["frames"],
                SEQUENCE_LENGTH,
            )

            frames = torch.stack(
                [load_frame_tensor(path) for path in selected_paths]
            )

            feature_chunks = []

            with torch.no_grad():
                for start in range(0, len(frames), FEATURE_BATCH_SIZE):
                    batch = frames[start : start + FEATURE_BATCH_SIZE].to(DEVICE)

                    features = model.cnn(batch)
                    features = features.reshape(features.size(0), 512)

                    feature_chunks.append(features.cpu())

            sequence_features = torch.cat(feature_chunks, dim=0)

            torch.save(
                {
                    "features": sequence_features,
                    "label": label,
                    "video_id": video_id,
                    "class_name": class_name,
                },
                cache_path,
            )

            cache_files.append(cache_path)

        except Exception as error:
            failed_groups.append(video_id)
            tqdm.write(f"Skipped {class_name}/{video_id}: {error}")

    print(
        f"\n{split_name.capitalize()} sequences cached: "
        f"{len(cache_files)}"
    )

    if failed_groups:
        print(f"Failed sequences: {len(failed_groups)}")

    return cache_files


class CachedVideoDataset(Dataset):
    def __init__(self, cache_files):
        self.cache_files = cache_files

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, index):
        data = torch.load(
            self.cache_files[index],
            map_location="cpu",
            weights_only=False,
        )

        return data["features"], data["label"]


def forward_from_features(model, features):
    lstm_output, _ = model.lstm(features)
    last_output = lstm_output[:, -1, :]
    return model.classifier(last_output)


def train_one_epoch(model, data_loader, loss_function, optimizer, epoch):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_videos = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch} - Training",
        unit="batch",
    )

    for features, labels in progress_bar:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = forward_from_features(model, features)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        current_batch_size = labels.size(0)

        total_loss += loss.item() * current_batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_videos += current_batch_size

        progress_bar.set_postfix(
            loss=f"{total_loss / total_videos:.4f}",
            accuracy=f"{correct_predictions / total_videos * 100:.2f}%",
        )

    return (
        total_loss / total_videos,
        correct_predictions / total_videos,
    )


def validate(model, data_loader, loss_function, epoch):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_videos = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch} - Validation",
        unit="batch",
    )

    with torch.no_grad():
        for features, labels in progress_bar:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = forward_from_features(model, features)
            loss = loss_function(outputs, labels)

            predictions = outputs.argmax(dim=1)
            current_batch_size = labels.size(0)

            total_loss += loss.item() * current_batch_size
            correct_predictions += (predictions == labels).sum().item()
            total_videos += current_batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total_videos:.4f}",
                accuracy=f"{correct_predictions / total_videos * 100:.2f}%",
            )

    return (
        total_loss / total_videos,
        correct_predictions / total_videos,
    )


def main():
    print("Using device:", DEVICE)
    print("Dataset:", DATASET_FOLDER)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    model = VideoCNNLSTM(
        hidden_size=256,
        num_layers=1,
        num_classes=2,
    ).to(DEVICE)

    for parameter in model.cnn.parameters():
        parameter.requires_grad = False

    train_groups = collect_video_groups(TRAIN_FOLDER)
    validation_groups = collect_video_groups(VALIDATION_FOLDER)

    print("\nDetected training video groups:", len(train_groups))
    print("Detected validation video groups:", len(validation_groups))

    print("\nPreparing CNN feature cache...")

    train_cache_files = create_feature_cache(
        model,
        TRAIN_FOLDER,
        "train_1000_videos",
    )

    validation_cache_files = create_feature_cache(
        model,
        VALIDATION_FOLDER,
        "validation_1000_videos",
    )

    if len(train_cache_files) == 0:
        raise ValueError("No valid training sequences found")

    if len(validation_cache_files) == 0:
        raise ValueError("No valid validation sequences found")

    train_dataset = CachedVideoDataset(train_cache_files)
    validation_dataset = CachedVideoDataset(validation_cache_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\nTraining sequences:", len(train_dataset))
    print("Validation sequences:", len(validation_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    loss_function = nn.CrossEntropyLoss()

    optimizer = Adam(
        list(model.lstm.parameters())
        + list(model.classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    best_validation_accuracy = 0.0

    WEIGHTS_FOLDER.mkdir(parents=True, exist_ok=True)

    print("\nStarting CNN + LSTM training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            epoch,
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            loss_function,
            epoch,
        )

        print(
            f"\nEpoch [{epoch}/{EPOCHS}]\n"
            f"Train Loss: {train_loss:.4f}\n"
            f"Train Accuracy: {train_accuracy * 100:.2f}%\n"
            f"Validation Loss: {validation_loss:.4f}\n"
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%\n"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_mapping": CLASS_MAPPING,
                    "validation_accuracy": validation_accuracy,
                    "epoch": epoch,
                    "sequence_length": SEQUENCE_LENGTH,
                    "image_size": IMAGE_SIZE,
                    "dataset": "1000_videos_frame_sequences",
                },
                WEIGHTS_PATH,
            )

            print("Best video model saved:", WEIGHTS_PATH)

    print(
        "\nVideo training completed."
        f"\nBest validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()