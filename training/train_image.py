from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.image_cnn import ImageCNN


# -------------------- SETTINGS --------------------

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.001

# Smaller balanced subsets for faster CPU training
MAX_TRAIN_IMAGES_PER_CLASS = 2000
MAX_VALIDATION_IMAGES_PER_CLASS = 500

TRAIN_FOLDER = PROJECT_ROOT / "dataset" / "image" / "train"
VALIDATION_FOLDER = PROJECT_ROOT / "dataset" / "image" / "validation"

WEIGHTS_FOLDER = PROJECT_ROOT / "backend" / "weights"
WEIGHTS_PATH = WEIGHTS_FOLDER / "image_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep this at 0 on Windows for stable multiprocessing
NUM_WORKERS = 0
RANDOM_SEED = 42


# -------------------- TRANSFORMS --------------------

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

validation_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# -------------------- BALANCED SUBSET --------------------

def create_balanced_subset(
    dataset: datasets.ImageFolder,
    images_per_class: int,
) -> Subset:
    random.seed(RANDOM_SEED)

    class_indices: dict[int, list[int]] = {}

    for index, label in enumerate(dataset.targets):
        if label not in class_indices:
            class_indices[label] = []

        class_indices[label].append(index)

    selected_indices: list[int] = []

    for label, indices in class_indices.items():
        random.shuffle(indices)

        number_to_select = min(
            images_per_class,
            len(indices),
        )

        selected_indices.extend(
            indices[:number_to_select]
        )

        print(
            f"Class {label}: selected "
            f"{number_to_select}/{len(indices)} images"
        )

    random.shuffle(selected_indices)

    return Subset(dataset, selected_indices)


# -------------------- DATA LOADERS --------------------

def create_data_loaders():
    if not TRAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_FOLDER}"
        )

    if not VALIDATION_FOLDER.exists():
        raise FileNotFoundError(
            f"Validation folder not found: {VALIDATION_FOLDER}"
        )

    full_train_dataset = datasets.ImageFolder(
        root=TRAIN_FOLDER,
        transform=train_transform,
    )

    full_validation_dataset = datasets.ImageFolder(
        root=VALIDATION_FOLDER,
        transform=validation_transform,
    )

    if full_train_dataset.class_to_idx != full_validation_dataset.class_to_idx:
        raise ValueError(
            "Train and validation class folders do not match.\n"
            f"Train classes: {full_train_dataset.class_to_idx}\n"
            f"Validation classes: {full_validation_dataset.class_to_idx}"
        )

    train_dataset = create_balanced_subset(
        full_train_dataset,
        MAX_TRAIN_IMAGES_PER_CLASS,
    )

    validation_dataset = create_balanced_subset(
        full_validation_dataset,
        MAX_VALIDATION_IMAGES_PER_CLASS,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")

    if len(validation_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    print("Class mapping:", full_train_dataset.class_to_idx)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    return (
        train_loader,
        validation_loader,
        full_train_dataset.class_to_idx,
    )


# -------------------- TRAINING --------------------

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Adam,
    epoch: int,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch} - Training",
        unit="batch",
    )

    for images, labels in progress_bar:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        predicted_labels = outputs.argmax(dim=1)
        current_batch_size = labels.size(0)

        total_loss += loss.item() * current_batch_size
        correct_predictions += (
            predicted_labels == labels
        ).sum().item()
        total_images += current_batch_size

        current_loss = total_loss / total_images
        current_accuracy = correct_predictions / total_images

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            accuracy=f"{current_accuracy * 100:.2f}%",
        )

    return (
        total_loss / total_images,
        correct_predictions / total_images,
    )


# -------------------- VALIDATION --------------------

def validate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    epoch: int,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch} - Validation",
        unit="batch",
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(images)
            loss = loss_function(outputs, labels)

            predicted_labels = outputs.argmax(dim=1)
            current_batch_size = labels.size(0)

            total_loss += loss.item() * current_batch_size
            correct_predictions += (
                predicted_labels == labels
            ).sum().item()
            total_images += current_batch_size

            current_loss = total_loss / total_images
            current_accuracy = correct_predictions / total_images

            progress_bar.set_postfix(
                loss=f"{current_loss:.4f}",
                accuracy=f"{current_accuracy * 100:.2f}%",
            )

    return (
        total_loss / total_images,
        correct_predictions / total_images,
    )


# -------------------- MAIN --------------------

def main():
    print("Using device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    (
        train_loader,
        validation_loader,
        class_mapping,
    ) = create_data_loaders()

    model = ImageCNN(num_classes=2).to(DEVICE)

    # Freeze all pretrained ResNet layers
    for parameter in model.cnn.parameters():
        parameter.requires_grad = False

    # Train only the final Real/Fake classifier
    for parameter in model.cnn.fc.parameters():
        parameter.requires_grad = True

    loss_function = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.cnn.fc.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = 0.0

    WEIGHTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                    "class_mapping": class_mapping,
                    "validation_accuracy": validation_accuracy,
                    "epoch": epoch,
                },
                WEIGHTS_PATH,
            )

            print("Best model saved:", WEIGHTS_PATH)

    print(
        "\nTraining completed."
        f"\nBest validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()