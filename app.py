"""
FastAPI microservice for real-time speech emotion recognition.
Handles audio file uploads and returns emotion predictions.
"""

import io
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import EmotionRecognitionModel

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="Real-time emotion classification from audio using fine-tuned Wav2Vec2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
emotion_model = EmotionRecognitionModel()


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    all_scores: dict


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health status."""
    return {
        "status": "healthy",
        "model": "wav2vec2-emotion",
        "device": str(emotion_model.device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(audio: UploadFile = File(...)):
    """
    Predict emotion from an uploaded audio file.
    
    Supports: WAV, MP3, FLAC, OGG formats.
    Max duration: 10 seconds.
    """
    if not audio.filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use WAV, MP3, FLAC, or OGG.",
        )

    try:
        contents = await audio.read()
        audio_array, sr = librosa.load(io.BytesIO(contents), sr=16000)

        if len(audio_array) / sr > 10:
            audio_array = audio_array[: sr * 10]  # Truncate to 10 seconds

        result = emotion_model.predict(audio_array, sampling_rate=sr)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
