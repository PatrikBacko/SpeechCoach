from pathlib import Path

import torch
import soundfile

from src.data_classes import AudioRecording


def load_recording(path: Path) -> AudioRecording:
    data, sample_rate = soundfile.read(str(path))
    return AudioRecording(samples=data, sample_rate=sample_rate, sentence=None)


def get_auto_device() -> torch.device:
    """Return the first available accelerator or CPU if none is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
