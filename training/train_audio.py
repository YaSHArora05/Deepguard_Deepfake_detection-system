from pathlib import Path
import random
import sys

import librosa
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.audio_cnn_lstm import AudioCNNLSTM

TRAIN_FOLDER = PROJECT_ROOT / "dataset" / "audio" / "train"
VALIDATION_FOLDER = PROJECT_ROOT / "dataset" / "audio" / "validation"
CACHE_FOLDER = PROJECT_ROOT / "dataset" / "audio" / "mel_cache"
WEIGHTS_FOLDER = PROJECT_ROOT / "backend" / "weights"
WEIGHTS_PATH = WEIGHTS_FOLDER / "audio_model.pth"

SAMPLE_RATE = 16000
DURATION_SECONDS = 4
MAX_AUDIO_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0003
NUM_WORKERS = 0
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAPPING = {"fake": 0, "real": 1}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def collect_audio_files(folder: Path):
    audio_items = []

    for class_name, label in CLASS_MAPPING.items():
        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(f"Missing folder: {class_folder}")

        for file_path in class_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
                audio_items.append(
                    {
                        "path": file_path,
                        "label": label,
                        "class_name": class_name,
                    }
                )

    return audio_items


def load_fixed_length_audio(file_path: Path) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) == 0:
        raise ValueError("Audio file is empty")

    if len(audio) > MAX_AUDIO_SAMPLES:
        audio = audio[:MAX_AUDIO_SAMPLES]
    elif len(audio) < MAX_AUDIO_SAMPLES:
        audio = np.pad(
            audio,
            (0, MAX_AUDIO_SAMPLES - len(audio)),
            mode="constant",
        )

    return audio


def create_mel_spectrogram(file_path: Path) -> torch.Tensor:
    audio = load_fixed_length_audio(file_path)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    mean = mel_spectrogram.mean()
    standard_deviation = mel_spectrogram.std()
    mel_spectrogram = (
        mel_spectrogram - mean
    ) / (standard_deviation + 1e-8)

    tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)
    return tensor.unsqueeze(0)


def create_mel_cache(source_folder: Path, split_name: str):
    audio_items = collect_audio_files(source_folder)
    split_cache_folder = CACHE_FOLDER / split_name
    split_cache_folder.mkdir(parents=True, exist_ok=True)

    cache_files = []
    failed_files = []

    progress_bar = tqdm(
        audio_items,
        desc=f"Caching {split_name} audio",
        unit="file",
    )

    for index, item in enumerate(progress_bar):
        audio_path = item["path"]
        class_name = item["class_name"]
        label = item["label"]

        class_cache_folder = split_cache_folder / class_name
        class_cache_folder.mkdir(parents=True, exist_ok=True)

        cache_path = class_cache_folder / f"{index}_{audio_path.stem}.pt"
        progress_bar.set_postfix(current=audio_path.name[:24])

        if cache_path.exists():
            cache_files.append(cache_path)
            continue

        try:
            mel_tensor = create_mel_spectrogram(audio_path)

            torch.save(
                {
                    "mel": mel_tensor,
                    "label": label,
                    "filename": audio_path.name,
                },
                cache_path,
            )

            cache_files.append(cache_path)
        except Exception as error:
            failed_files.append(audio_path.name)
            tqdm.write(f"Skipped {audio_path.name}: {error}")

    print(f"\n{split_name.capitalize()} cache created: {len(cache_files)} files")

    if failed_files:
        print(f"Failed files: {len(failed_files)}")

    return cache_files


class CachedAudioDataset(Dataset):
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
        return data["mel"], data["label"]


def train_one_epoch(model, data_loader, loss_function, optimizer, epoch):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_files = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch} - Training",
        unit="batch",
    )

    for audio_tensors, labels in progress_bar:
        audio_tensors = audio_tensors.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(audio_tensors)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        current_batch_size = labels.size(0)
        total_loss += loss.item() * current_batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_files += current_batch_size

        progress_bar.set_postfix(
            loss=f"{total_loss / total_files:.4f}",
            accuracy=f"{correct_predictions / total_files * 100:.2f}%",
        )

    return total_loss / total_files, correct_predictions / total_files


def validate(model, data_loader, loss_function, epoch):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_files = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch} - Validation",
        unit="batch",
    )

    with torch.no_grad():
        for audio_tensors, labels in progress_bar:
            audio_tensors = audio_tensors.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(audio_tensors)
            loss = loss_function(outputs, labels)
            predictions = outputs.argmax(dim=1)
            current_batch_size = labels.size(0)

            total_loss += loss.item() * current_batch_size
            correct_predictions += (predictions == labels).sum().item()
            total_files += current_batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total_files:.4f}",
                accuracy=f"{correct_predictions / total_files * 100:.2f}%",
            )

    return total_loss / total_files, correct_predictions / total_files


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Using device:", DEVICE)
    print("Sample rate:", SAMPLE_RATE)
    print("Duration:", DURATION_SECONDS, "seconds")
    print("Mel bands:", N_MELS)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    print("\nPreparing Mel-spectrogram cache...")

    train_cache_files = create_mel_cache(TRAIN_FOLDER, "train")
    validation_cache_files = create_mel_cache(
        VALIDATION_FOLDER,
        "validation",
    )

    if len(train_cache_files) == 0:
        raise ValueError("No valid training audio files found")

    if len(validation_cache_files) == 0:
        raise ValueError("No valid validation audio files found")

    train_dataset = CachedAudioDataset(train_cache_files)
    validation_dataset = CachedAudioDataset(validation_cache_files)

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

    print("\nTraining audio files:", len(train_dataset))
    print("Validation audio files:", len(validation_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    model = AudioCNNLSTM(
        hidden_size=128,
        num_classes=2,
    ).to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    best_validation_accuracy = 0.0
    WEIGHTS_FOLDER.mkdir(parents=True, exist_ok=True)

    print("\nStarting CNN + LSTM audio training...\n")

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
            f"Validation Accuracy: {validation_accuracy * 100:.2f}%\n"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_mapping": CLASS_MAPPING,
                    "validation_accuracy": validation_accuracy,
                    "epoch": epoch,
                    "sample_rate": SAMPLE_RATE,
                    "duration_seconds": DURATION_SECONDS,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                },
                WEIGHTS_PATH,
            )

            print("Best audio model saved:", WEIGHTS_PATH)

    print(
        "\nAudio training completed."
        f"\nBest validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()