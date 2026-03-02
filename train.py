"""
Training script for Wav2Vec2 Speech Emotion Recognition.
Fine-tunes wav2vec2-base on the RAVDESS dataset.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import librosa
import glob


EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"
]


class RAVDESSDataset(Dataset):
    """PyTorch Dataset for RAVDESS audio emotion data."""

    def __init__(self, file_paths, labels, feature_extractor, max_length=160000):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.file_paths[idx], sr=16000)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_ravdess_data(data_dir: str):
    """Load file paths and labels from RAVDESS dataset directory."""
    file_paths = []
    labels = []

    audio_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)

    for filepath in audio_files:
        filename = os.path.basename(filepath)
        parts = filename.split("-")
        if len(parts) >= 3:
            emotion_code = int(parts[2]) - 1  # RAVDESS is 1-indexed
            if emotion_code < len(EMOTION_LABELS):
                file_paths.append(filepath)
                labels.append(emotion_code)

    return file_paths, labels


def compute_metrics(pred):
    """Compute accuracy and F1 score for evaluation."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {"accuracy": accuracy, "f1": f1}


def train(data_dir: str, output_dir: str = "./trained_model", epochs: int = 10):
    """Fine-tune Wav2Vec2 on RAVDESS dataset."""
    print("Loading RAVDESS dataset...")
    file_paths, labels = load_ravdess_data(data_dir)
    print(f"Found {len(file_paths)} audio samples across {len(set(labels))} emotions")

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Initialize model and feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(EMOTION_LABELS),
    )

    # Create datasets
    train_dataset = RAVDESSDataset(train_paths, train_labels, feature_extractor)
    val_dataset = RAVDESSDataset(val_paths, val_labels, feature_extractor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to RAVDESS dataset")
    parser.add_argument("--output_dir", type=str, default="./trained_model")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train(args.data_dir, args.output_dir, args.epochs)
