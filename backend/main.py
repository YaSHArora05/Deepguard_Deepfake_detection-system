from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_file


app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Deepfake Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was selected",
        )

    extension = Path(file.filename).suffix.lower()

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
    }

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
        )

    unique_filename = f"{uuid.uuid4()}{extension}"
    saved_file_path = UPLOAD_FOLDER / unique_filename

    try:
        with saved_file_path.open("wb") as output_file:
            shutil.copyfileobj(
                file.file,
                output_file,
            )

        result = predict_file(str(saved_file_path))

        return {
            "filename": file.filename,
            **result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )

    finally:
        await file.close()

        if saved_file_path.exists():
            saved_file_path.unlink()