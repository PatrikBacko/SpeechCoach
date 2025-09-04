from dataclasses import dataclass
import numpy as np
from scipy.signal import resample
import scipy.io.wavfile
import io


@dataclass
class AudioRecording:
    samples: np.ndarray
    sample_rate: int
    sentence: str | None

    def resample(self, target_sample_rate: int) -> "AudioRecording":
        if self.sample_rate == target_sample_rate:
            return self
        num_samples = int(len(self.samples) * target_sample_rate / self.sample_rate)
        resampled = resample(self.samples, num_samples)
        return AudioRecording(
            samples=resampled.astype(self.samples.dtype),
            sample_rate=target_sample_rate,
            sentence=self.sentence
        )
    
    def to_json(self) -> dict:
        return {
            "samples": self.samples.tolist(),
            "sample_rate": self.sample_rate,
            "sentence": self.sentence,
            "dtype": str(self.samples.dtype)
        }
    
    @classmethod
    def from_json(cls, data: dict) -> "AudioRecording":
        return cls(
            samples=np.array(data["samples"], dtype=data.get("dtype", "float32")),
            sample_rate=data["sample_rate"],
            sentence=data.get("sentence")
        )
    
    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        samples_int16 = (self.samples * 32767).astype(np.int16) if self.samples.dtype != np.int16 else self.samples
        scipy.io.wavfile.write(buf, self.sample_rate, samples_int16)
        return buf.getvalue()

    @classmethod
    def from_wav_bytes(cls, wav_bytes: bytes, sentence: str | None = None) -> "AudioRecording":
        buf = io.BytesIO(wav_bytes)
        sample_rate, samples = scipy.io.wavfile.read(buf)
        # Normalize to float32 if samples are int16
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32767
        return cls(samples=samples, sample_rate=sample_rate, sentence=sentence)


@dataclass
class Transcription:
    sentence: str
    tokens_with_probs: list[tuple[str, float]]


@dataclass
class TargetSentence:
    sentence: str
    tokens: list[str]


@dataclass
class Mistake:
    start_idx: int
    end_idx: int
    mistaken_text: str

    def to_json(self) -> dict:
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "mistaken_text": self.mistaken_text
        }
    
    @classmethod
    def from_json(cls, data: dict) -> "Mistake":
        return cls(
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            mistaken_text=data["mistaken_text"]
        )
