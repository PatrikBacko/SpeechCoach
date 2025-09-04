from abc import ABC, abstractmethod
from typing import Literal

from kokoro import KPipeline
import soundfile as sf
import torch

from src.data_classes import AudioRecording
from src.utils import get_auto_device

class TTSModel(ABC):
    @abstractmethod
    def synthesize(self, sentence: str) -> AudioRecording: # audio recording
        ...


class KokoroTTSModel(TTSModel):
    voices = {
        'en-us': 'af_heart',
        'en-gb': 'bf_isabella',
    }
    def __init__(
        self,
        device: torch.device | str | None = None,
        language: Literal['en-us', 'en-gb'] = 'en-us',
    ):
        self._pipeline = KPipeline(
            lang_code=language,
            device= device or get_auto_device(),
            repo_id='hexgrad/Kokoro-82M',
        )
        self._voice = self.voices[language]

    def synthesize(
        self,
        sentence: str,
        speed: float = 0.75
    ) -> AudioRecording:
        generator = self._pipeline(
            sentence,
            voice=self._voice,
            speed=speed,
        )
        audio = next(generator).audio.detach().cpu().numpy()
        return AudioRecording(audio, 24000, sentence)
