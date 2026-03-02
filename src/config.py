"""
Configuration loader for Speech Emotion Recognition.
"""

import os
import yaml
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    pretrained_model: str = "facebook/wav2vec2-base"
    num_labels: int = 7
    max_audio_length: int = 160000  # 10 seconds at 16kHz
    sampling_rate: int = 16000


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    test_size: float = 0.2
    seed: int = 42
    fp16: bool = True


@dataclass
class APIConfig:
    """FastAPI service configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_audio_duration: int = 10
    allowed_formats: tuple = (".wav", ".mp3", ".flac", ".ogg")


@dataclass
class AppConfig:
    """Application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    output_dir: str = "./trained_model"
    log_dir: str = "./logs"


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from YAML file."""
    config = AppConfig()

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "api" in data:
            config.api = APIConfig(**data["api"])

    return config
