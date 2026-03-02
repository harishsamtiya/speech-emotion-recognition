"""
Unit tests for Speech Emotion Recognition components.
"""

import pytest
import numpy as np
from src.preprocessing import (
    pad_or_truncate, augment_audio, get_audio_stats,
    extract_mel_spectrogram, EMOTION_LABELS,
)


class TestPreprocessing:
    """Tests for audio preprocessing utilities."""

    def test_pad_short_audio(self):
        short_audio = np.random.randn(8000)
        result = pad_or_truncate(short_audio, max_length=16000)
        assert len(result) == 16000

    def test_truncate_long_audio(self):
        long_audio = np.random.randn(320000)
        result = pad_or_truncate(long_audio, max_length=160000)
        assert len(result) == 160000

    def test_exact_length_unchanged(self):
        audio = np.random.randn(160000)
        result = pad_or_truncate(audio, max_length=160000)
        assert np.array_equal(audio, result)

    def test_augment_adds_noise(self):
        audio = np.zeros(16000)
        augmented = augment_audio(audio, noise_factor=0.01)
        assert not np.array_equal(audio, augmented)
        assert len(augmented) == len(audio)

    def test_mel_spectrogram_shape(self):
        audio = np.random.randn(16000)
        mel = extract_mel_spectrogram(audio, sr=16000, n_mels=128)
        assert mel.shape[0] == 128

    def test_audio_stats(self):
        audio = np.random.randn(16000)
        stats = get_audio_stats(audio, sr=16000)
        assert "duration_seconds" in stats
        assert "rms_energy" in stats
        assert stats["duration_seconds"] == pytest.approx(1.0, abs=0.01)

    def test_emotion_labels_count(self):
        assert len(EMOTION_LABELS) == 7
        assert "happy" in EMOTION_LABELS
        assert "angry" in EMOTION_LABELS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
