"""
Audio preprocessing utilities for Speech Emotion Recognition.
Handles loading, resampling, augmentation, and feature extraction.
"""

import numpy as np
import librosa
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"
]


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load and resample audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        logger.debug(f"Loaded {file_path}: {len(audio)/sr:.2f}s at {sr}Hz")
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def pad_or_truncate(audio: np.ndarray, max_length: int = 160000) -> np.ndarray:
    """Pad short audio or truncate long audio to fixed length."""
    if len(audio) > max_length:
        return audio[:max_length]
    elif len(audio) < max_length:
        padding = max_length - len(audio)
        return np.pad(audio, (0, padding), mode="constant")
    return audio


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    noise_factor: float = 0.005,
    pitch_shift: Optional[int] = None,
    time_stretch: Optional[float] = None,
) -> np.ndarray:
    """Apply audio data augmentation."""
    augmented = audio.copy()

    # Add white noise
    if noise_factor > 0:
        noise = np.random.randn(len(augmented)) * noise_factor
        augmented = augmented + noise

    # Pitch shifting
    if pitch_shift is not None:
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=pitch_shift)

    # Time stretching
    if time_stretch is not None:
        augmented = librosa.effects.time_stretch(augmented, rate=time_stretch)

    return augmented


def extract_mel_spectrogram(
    audio: np.ndarray, sr: int = 16000, n_mels: int = 128
) -> np.ndarray:
    """Extract Mel spectrogram features from audio."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def get_audio_stats(audio: np.ndarray, sr: int = 16000) -> dict:
    """Get statistical features from audio signal."""
    return {
        "duration_seconds": len(audio) / sr,
        "rms_energy": float(np.sqrt(np.mean(audio**2))),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
        "sample_count": len(audio),
    }
