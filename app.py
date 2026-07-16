from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import pathlib

from predict import predict_image

os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_extension(filename: str) -> str:
    """Return a safe extension for an uploaded file."""
    ext = pathlib.Path(filename).suffix.lower()

    # Allow common image extensions only
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        return ext

    # Default to a common image extension so load_img can still try
    return ".img"


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    ext = _safe_extension(image.filename)
    tmp_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join("uploads", tmp_name)

    try:
        # Save upload to disk for Keras loader compatibility
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = predict_image(file_path)
        return result

    except Exception:
        # Avoid leaking internal details to the client
        raise HTTPException(status_code=500, detail="Prediction failed")

    finally:
        # Cleanup uploaded temp file even if prediction fails
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

