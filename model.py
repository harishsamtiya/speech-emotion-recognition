"""
Real-Time Speech Emotion Recognition
======================================
Fine-tuned Wav2Vec2 model for classifying 7 emotion categories
from audio input. Deployed as a FastAPI microservice.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


# Emotion labels from RAVDESS dataset
EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"
]


class EmotionRecognitionModel:
    """Wav2Vec2-based speech emotion recognition model."""

    def __init__(self, model_path: str = None, num_labels: int = 7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )

        if model_path:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels
            )
        else:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "facebook/wav2vec2-base", num_labels=num_labels
            )

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, audio_array, sampling_rate: int = 16000):
        """Preprocess raw audio for model input."""
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=160000,  # 10 seconds max
            truncation=True,
        )
        return inputs.input_values.to(self.device)

    def predict(self, audio_array, sampling_rate: int = 16000) -> dict:
        """Predict emotion from audio array."""
        input_values = self.preprocess(audio_array, sampling_rate)

        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        predicted_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_idx].item()

        return {
            "emotion": EMOTION_LABELS[predicted_idx],
            "confidence": round(confidence, 4),
            "all_scores": {
                label: round(probs[0][i].item(), 4)
                for i, label in enumerate(EMOTION_LABELS)
            },
        }
