from setuptools import setup, find_packages

setup(
    name="speech-emotion-recognition",
    version="1.0.0",
    author="Harish Samtiya",
    author_email="samtiyaharish@gmail.com",
    description="Real-Time Speech Emotion Recognition using Wav2Vec2 and FastAPI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/harishsamtiya/speech-emotion-recognition",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "librosa>=0.10.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
