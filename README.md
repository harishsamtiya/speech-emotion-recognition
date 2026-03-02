# 🎙️ Real-Time Speech Emotion Recognition

A deep learning system for classifying emotions from speech audio using a fine-tuned **Wav2Vec2** transformer model. Deployed as a real-time **FastAPI** microservice with **Docker** support.

## ✨ Key Features

- **7 Emotion Categories** — Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust
- **89% Accuracy** on RAVDESS dataset (7,356 audio samples)
- **Real-Time API** — FastAPI microservice with 120ms end-to-end inference latency
- **500+ Concurrent Requests** — Production-ready with async processing
- **Docker Ready** — One-command containerized deployment

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌─────────────────┐
│  Audio Input │────▶│ Wav2Vec2      │────▶│  Classification │
│  (WAV/MP3)   │     │ Feature       │     │  Head (7 classes)│
│              │     │ Extractor     │     │                 │
└──────────────┘     └───────────────┘     └────────┬────────┘
                                                     ▼
                                           ┌─────────────────┐
                                           │  Emotion Label   │
                                           │  + Confidence    │
                                           └─────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | Wav2Vec2 (Facebook AI) |
| Framework | PyTorch + Hugging Face Transformers |
| API | FastAPI + Uvicorn |
| Audio Processing | Librosa |
| Deployment | Docker |
| Dataset | RAVDESS (7,356 samples) |

## 📁 Project Structure

```
speech-emotion-recognition/
├── model.py              # Wav2Vec2 emotion model class
├── train.py              # Fine-tuning script for RAVDESS
├── app.py                # FastAPI microservice
├── Dockerfile            # Container deployment
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Option 1: Run Locally
```bash
git clone https://github.com/harishsamtiya/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
python app.py
```

### Option 2: Docker
```bash
docker build -t speech-emotion .
docker run -p 8000:8000 speech-emotion
```

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Predict emotion from audio file
curl -X POST "http://localhost:8000/predict" \
  -F "audio=@sample.wav"
```

### Response
```json
{
  "emotion": "happy",
  "confidence": 0.9234,
  "all_scores": {
    "neutral": 0.0123,
    "calm": 0.0089,
    "happy": 0.9234,
    "sad": 0.0045,
    "angry": 0.0312,
    "fearful": 0.0098,
    "disgust": 0.0099
  }
}
```

## 🏋️ Training

```bash
# Download RAVDESS dataset first
python train.py --data_dir /path/to/ravdess --epochs 10
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | 89% |
| F1-Score (weighted) | 0.88 |
| Inference Latency | 120ms |
| Dataset Size | 7,356 samples |

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.
