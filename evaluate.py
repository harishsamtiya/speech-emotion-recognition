"""
Model evaluation script for Speech Emotion Recognition.
Generates classification reports, confusion matrix, and per-class metrics.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from model import EmotionRecognitionModel
from src.preprocessing import load_audio, EMOTION_LABELS
import glob
import json


def evaluate_model(model_path: str, data_dir: str, output_dir: str = "./evaluation"):
    """Run full evaluation pipeline on test data."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model = EmotionRecognitionModel(model_path=model_path)

    print("Loading test data...")
    file_paths, true_labels = load_test_data(data_dir)
    print(f"Found {len(file_paths)} test samples")

    predictions = []
    confidences = []

    print("Running predictions...")
    for i, path in enumerate(file_paths):
        audio, sr = load_audio(path)
        result = model.predict(audio, sr)
        pred_idx = EMOTION_LABELS.index(result["emotion"])
        predictions.append(pred_idx)
        confidences.append(result["confidence"])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(file_paths)}")

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    report = classification_report(
        true_labels, predictions,
        target_names=EMOTION_LABELS,
        output_dict=True,
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Avg Confidence: {np.mean(confidences):.4f}")

    # Save classification report
    report_text = classification_report(
        true_labels, predictions, target_names=EMOTION_LABELS
    )
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    # Save metrics JSON
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "avg_confidence": float(np.mean(confidences)),
        "per_class": report,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(true_labels, predictions, output_dir)

    print(f"\nEvaluation saved to {output_dir}")
    return metrics


def load_test_data(data_dir: str):
    """Load test data paths and labels from RAVDESS format."""
    file_paths = []
    labels = []
    audio_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)

    for filepath in audio_files:
        filename = os.path.basename(filepath)
        parts = filename.split("-")
        if len(parts) >= 3:
            emotion_code = int(parts[2]) - 1
            if emotion_code < len(EMOTION_LABELS):
                file_paths.append(filepath)
                labels.append(emotion_code)

    return file_paths, labels


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
    )
    plt.title("Confusion Matrix (Normalized)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Confusion matrix saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SER Model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./evaluation")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, args.output_dir)
