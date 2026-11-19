# DeepGuard: Multimodal Deepfake Detection

DeepGuard checks whether an **image, video or audio clip** is real or AI-generated. You upload a file in a React web app. A FastAPI backend works out the media type, preprocesses the file and runs the matching PyTorch model. It returns a **REAL / FAKE** prediction with a confidence score and class probabilities.

| Media | Model | Input to the model |
|-------|-------|--------------------|
| Image | ResNet18 CNN (ImageNet-pretrained, new 2-class head) | 224×224 RGB image |
| Video | ResNet18 frame encoder + LSTM | 20 evenly spaced frames at 112×112 |
| Audio | 2-layer CNN + LSTM | 128-band log-Mel spectrogram at 16 kHz |

---

## Features

- **One upload endpoint for all media.** The backend picks the model from the file extension.
- **Drag-and-drop web UI** with a result card showing the prediction, confidence bar and real/fake probabilities.
- **Video decoding fallback.** If OpenCV can't open a video, it is re-encoded to H.264 with a bundled ffmpeg and then read.
- **Graceful handling of untrained models.** If a model's weights are missing, the API returns `"prediction": "unavailable"` instead of crashing.
- **Training scripts for every modality.** They cache preprocessed features (Mel spectrograms and frame embeddings), so later runs are fast even on a CPU.
- **Uploads are temporary.** Each upload is saved under a random UUID name and deleted right after prediction.

---

## Tech stack

**Frontend:** React 19, TypeScript, Vite, Oxlint
**Backend:** FastAPI, Uvicorn, PyTorch, Torchvision, OpenCV, Librosa, Pillow, imageio-ffmpeg
**Training:** PyTorch, tqdm

---

## Project structure

```
anti-ai-deepfake-detection/
├── backend/
│   ├── main.py                 # FastAPI app: GET / and POST /predict
│   ├── predict.py              # Loads weights and runs inference
│   ├── preprocess.py           # Image / video / audio → tensors
│   ├── models/
│   │   ├── image_cnn.py        # ResNet18 classifier
│   │   ├── video_cnn_lstm.py   # ResNet18 + LSTM
│   │   └── audio_cnn_lstm.py   # CNN + LSTM over Mel spectrograms
│   ├── weights/                # *.pth files go here (git-ignored, create after training)
│   ├── test_*.py               # Quick manual sanity-check scripts
│   ├── sample.jpg              # Sample image used by the test scripts
│   └── requirements.txt
├── training/
│   ├── train_image.py
│   ├── train_video.py
│   └── train_audio.py
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── components/         # FileUpload, ResultCard
│       └── services/api.ts     # Calls the backend
└── dataset/                    # Not committed. See "Dataset layout"
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 20+ and npm
- (Optional) An NVIDIA GPU with CUDA. Everything also runs on a CPU, just more slowly.

### 1. Backend

```bash
cd backend
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install imageio-ffmpeg tqdm   # used by preprocess.py and the training scripts

uvicorn main:app --reload
```

The API runs at **http://127.0.0.1:8000**. Interactive docs are at http://127.0.0.1:8000/docs.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**.

> The backend only accepts browser requests from `http://localhost:5173` (CORS), and the frontend calls `http://127.0.0.1:8000`. If you change either port, update `allow_origins` in `backend/main.py` and `API_URL` in `frontend/src/services/api.ts`.

### 3. Add model weights

The API looks for these files:

```
backend/weights/image_model.pth
backend/weights/video_model.pth
backend/weights/audio_model.pth
```

Weight files are git-ignored, so a fresh clone has none. Train them (below) or copy them in. Each checkpoint must be a dict with a `model_state_dict` key, which is the format the training scripts save. Until a model's weights exist, requests for that media type return `"prediction": "unavailable"`.

---

## Dataset layout

Put datasets in a `dataset/` folder at the project root. Each split needs `fake/` and `real/` subfolders.

```
dataset/
├── image/
│   ├── train/{fake,real}/*.jpg|png
│   └── validation/{fake,real}/*.jpg|png
├── audio/
│   ├── train/{fake,real}/*.wav|mp3|flac|ogg
│   └── validation/{fake,real}/*.wav|mp3|flac|ogg
└── video/
    └── 1000_videos/
        ├── train/{fake,real}/<video_id>_<frame_no>.png
        └── validation/{fake,real}/<video_id>_<frame_no>.png
```

Video training uses **pre-extracted frames**, not raw video files. Frames are grouped into clips by the part of the file name before the last underscore, and ordered by the number after it (for example `clip042_0007.png`).

---

## Training

Run the scripts from the project root:

```bash
python training/train_image.py
python training/train_video.py
python training/train_audio.py
```

Each script keeps the checkpoint with the best validation accuracy and saves it to `backend/weights/`.

| Script | What is trained | Key defaults |
|--------|-----------------|--------------|
| `train_image.py` | Only the final classifier layer (ResNet18 backbone frozen) | 2 epochs, batch 64, lr 1e-3, balanced subset of 2,000 train / 500 val images per class |
| `train_video.py` | LSTM + classifier on cached ResNet18 frame features | 15 epochs, batch 8, lr 3e-4, 20 frames per clip |
| `train_audio.py` | Full CNN + LSTM | 10 epochs, batch 16, lr 3e-4, clips padded or cut to 4 s |

Settings are constants at the top of each script. The image limits are set low so training finishes quickly on a CPU; raise `EPOCHS` and the per-class caps for better accuracy. Feature caches are written to `dataset/video/frame_feature_cache/` and `dataset/audio/mel_cache/`. Delete them if you change the preprocessing.

---

## API reference

### `GET /`

Health check.

```json
{ "message": "Deepfake Detection API is running" }
```

### `POST /predict`

Send a multipart form with a single `file` field.

**Supported extensions:** `.jpg .jpeg .png` · `.mp4 .avi .mov .mkv` · `.wav .mp3 .flac .ogg`

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@backend/sample.jpg"
```

**Success response**

```json
{
  "filename": "sample.jpg",
  "file_type": "image",
  "prediction": "fake",
  "confidence": 0.9134,
  "fake_probability": 0.9134,
  "real_probability": 0.0866
}
```

**Model not trained yet**

```json
{
  "filename": "clip.mp4",
  "file_type": "video",
  "prediction": "unavailable",
  "message": "Video model is not trained yet"
}
```

**Errors:** `400` for a missing or unsupported file, or an unreadable video or audio file; `500` if inference fails.

---

## Sanity-check scripts

From inside `backend/`:

```bash
python test_preprocess.py     # preprocess sample.jpg and print the tensor shape
python test_image_model.py    # forward pass through ImageCNN
python test_video_model.py    # forward pass with a random video tensor
python test_audio_model.py    # forward pass with a random spectrogram
python test_predict.py        # full pipeline on sample.jpg
```

These are quick print-based checks, not a pytest suite.

---

## Known limitations

- Class indices are fixed as `0 = fake`, `1 = real`. The image script takes its mapping from `ImageFolder`, which sorts folder names alphabetically, so keep the folders named exactly `fake` and `real`.
- Audio training uses fixed 4-second clips with `n_fft=1024`. Inference uses the whole file with librosa's default `n_fft`. Matching the two would likely improve audio accuracy.
- The video model was trained on frame images. At inference it samples frames from the uploaded video, so results depend on how similar those frames are to the training frames (for example, face crops versus full frames).
- Accuracy depends entirely on your dataset. Treat the output as a signal for review, not proof.

---

## License

No license has been chosen yet. Add a `LICENSE` file before publishing or accepting contributions.
