from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

IMAGE_SIZE = 224
VIDEO_IMAGE_SIZE = 112
SEQUENCE_LENGTH = 20

SAMPLE_RATE = 16000
N_MELS = 128


image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


video_transform = transforms.Compose(
    [
        transforms.Resize(
            (VIDEO_IMAGE_SIZE, VIDEO_IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def get_file_type(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension in VIDEO_EXTENSIONS:
        return "video"

    if extension in AUDIO_EXTENSIONS:
        return "audio"

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def preprocess_image(file_path: str) -> torch.Tensor:
    image = Image.open(file_path).convert("RGB")
    image_tensor = image_transform(image)

    return image_tensor.unsqueeze(0)


def convert_video_to_h264(file_path: str) -> str:
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    temporary_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )

    converted_path = temporary_file.name
    temporary_file.close()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        file_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        converted_path,
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    except subprocess.CalledProcessError as error:
        Path(converted_path).unlink(
            missing_ok=True
        )

        raise ValueError(
            "Could not convert the video file"
        ) from error

    return converted_path


def open_video(file_path: str):
    video = cv2.VideoCapture(file_path)

    if video.isOpened():
        return video, None

    video.release()

    converted_path = convert_video_to_h264(
        file_path
    )

    converted_video = cv2.VideoCapture(
        converted_path
    )

    if not converted_video.isOpened():
        converted_video.release()

        Path(converted_path).unlink(
            missing_ok=True
        )

        raise ValueError(
            "Could not decode the video file"
        )

    return converted_video, converted_path


def preprocess_video(file_path: str) -> torch.Tensor:
    video, converted_path = open_video(
        file_path
    )

    try:
        total_frames = int(
            video.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            raise ValueError(
                "Video contains no readable frames"
            )

        frame_indexes = np.linspace(
            0,
            total_frames - 1,
            SEQUENCE_LENGTH,
            dtype=int,
        )

        frames = []

        for frame_index in frame_indexes:
            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            success, frame = video.read()

            if not success:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frame = Image.fromarray(frame)
            frame = video_transform(frame)

            frames.append(frame)

    finally:
        video.release()

        if converted_path is not None:
            Path(converted_path).unlink(
                missing_ok=True
            )

    if len(frames) == 0:
        raise ValueError(
            "No valid frames could be extracted"
        )

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(
            frames[-1].clone()
        )

    frames = frames[:SEQUENCE_LENGTH]

    video_tensor = torch.stack(frames)

    return video_tensor.unsqueeze(0)


def preprocess_audio(file_path: str) -> torch.Tensor:
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(audio) == 0:
        raise ValueError(
            "Audio file is empty"
        )

    mel_spectrogram = (
        librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
        )
    )

    mel_spectrogram = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
    )

    mean = mel_spectrogram.mean()
    standard_deviation = (
        mel_spectrogram.std()
    )

    mel_spectrogram = (
        mel_spectrogram - mean
    ) / (standard_deviation + 1e-8)

    audio_tensor = torch.tensor(
        mel_spectrogram,
        dtype=torch.float32,
    )

    audio_tensor = (
        audio_tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return audio_tensor


def preprocess_file(
    file_path: str,
) -> tuple[str, torch.Tensor]:
    file_type = get_file_type(file_path)

    if file_type == "image":
        tensor = preprocess_image(
            file_path
        )

    elif file_type == "video":
        tensor = preprocess_video(
            file_path
        )

    else:
        tensor = preprocess_audio(
            file_path
        )

    return file_type, tensor