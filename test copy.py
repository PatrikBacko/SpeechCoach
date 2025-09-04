import re
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import override

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch
import numpy as np
import soundfile as sf

# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
# model.config.forced_decoder_ids = None

# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
# input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# predicted_ids = model.generate(input_features, num_beams=100, output_logits=True, return_dict_in_generate=True, language='en')

# transcription = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)

# pred = [p for p in predicted_ids.sequences.squeeze() if p.item() not in processor.tokenizer.added_tokens_decoder.keys()]

# [torch.softmax(logits, dim=1)[:,pred[i]] for i, logits in enumerate(predicted_ids.logits[:len(pred)])]


@dataclass
class AudioRecording:
    recording: np.array
    sample_rate: int
    sentence: str | None # ??


@dataclass
class Transcription:
    sentence: str
    tokens_with_probs: list[tuple[str, float]]


@dataclass
class TargetSentence:
    sentence: str
    tokens: list[str]


class RecognitionModel(ABC):
    @abstractmethod
    def transcribe(self, recording: AudioRecording) -> Transcription:
        ...

    @abstractmethod
    def tokenize_sentence(self, sentence: str) -> TargetSentence:
        ...


class WhisperRecognitionModel(RecognitionModel):
    def __init__(
        self,
        processor: WhisperProcessor,
        model: WhisperForConditionalGeneration,
        num_beams: int = 1
    ):
        super().__init__()
        self._processor = processor
        self._model = model
        self._num_beams = num_beams

    @classmethod
    def create_from_name(cls, name, num_beams: int = 1):
        processor = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name)
        return cls(processor, model, num_beams)

    @override
    def transcribe(self, recording: AudioRecording) -> Transcription:
        input_features = self._processor(recording.recording, sampling_rate=recording.sample_rate, return_tensors="pt").input_features
        predicted_ids = self._model.generate(input_features, num_beams=self._num_beams, output_logits=True, return_dict_in_generate=True)
        transcription = self._processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]
        print(transcription)
        pred = [p for p in predicted_ids.sequences.squeeze() if p.item() not in self._processor.tokenizer.added_tokens_decoder.keys()]
        probs = [torch.softmax(logits, dim=1)[:,pred[i]] for i, logits in enumerate(predicted_ids.logits[:len(pred)])]
        return Transcription(transcription, list(zip(self._processor.batch_decode(pred), probs)))
    
    @override
    def tokenize_sentence(self, sentence: str) -> TargetSentence:
        tokens = self._processor.tokenizer.tokenize(sentence)
        return TargetSentence(sentence, tokens)
        

class TTSModel(ABC):
    @abstractmethod
    def generate_audio(self, text: str) -> AudioRecording: # audio recording
        ...


@dataclass
class Mistake:
    start_idx: int
    end_idx: int
    mistaken_text: str


class LLMModel(ABC):
    @abstractmethod
    def generate_target_sentence(self) -> str:
        # TODO
        ...

    @abstractmethod
    def generate_suggestions(self, mistake: Mistake, sentence: str) -> str:
        # TODO
        ...


class HuggingFaceLLMModel(LLMModel):
    def __init__(
        self
    ):
        super().__init__()

    @classmethod
    def create_from_name(cls, name):
        pass

class MistakeFinder(ABC):
    @abstractmethod
    def find_mistakes(
        self,
        transcription: Transcription,
        target_sentence: str,
        target_tokens: str | None = None, # ???
    ) -> list[Mistake]:
        pass


class MistakeFinderBasic(MistakeFinder):
    @override
    def find_mistakes(
        self,
        transcription: Transcription,
        target_sentence: str,
        target_tokens: str | None = None, # ???
    ) -> list[Mistake]:
        target_sent_words = self._split_with_indices(target_sentence)
        pred_sent_words = self._split_with_indices(transcription.sentence)

        mistakes = []

        for (target_word, target_idx), (pred_word, pred_idx) in zip(target_sent_words, pred_sent_words):
            if target_word != pred_word:
                mistakes.append(
                    Mistake(target_idx, target_idx + len(target_word), pred_word)  
                )

        return mistakes

    def _split_with_indices(
        self,
        text: str,
        delimiter: str = ' '
    ) -> list[tuple[str, int]]:
        pattern = re.compile(r'[^' + re.escape(delimiter) + r']+')
        return [(m.group(), m.start()) for m in pattern.finditer(text)]


class PronounciationScorer:
    def __init__(
        self,
        recognition_model: RecognitionModel,
        tts_model : TTSModel,
        llm_model: LLMModel,

        mistake_finder: MistakeFinder,
    ):
        self._recognition_model = recognition_model
        self._tts_model = tts_model
        self._llm_model = llm_model

        self._mistake_finder = mistake_finder

    def find_mistakes(
        self,
        sentence: str,
        recording: AudioRecording,
    ) -> list[Mistake]:
        transcription = self._recognition_model.transcribe(recording)
        # print(transcription)
        return self._mistake_finder.find_mistakes(transcription, sentence, None)
    
    def generetare_suggestion(
        self,
        mistake: Mistake
    ):
        return self._llm_model.generate_suggestions(mistake)
        
    def generate_target_sentence(
        self
    ) -> str:
        return self._llm_model.generate_target_sentence()


def load_recording(path: Path) -> AudioRecording:
    data, sample_rate = sf.read(str(path))
    return AudioRecording(recording=data, sample_rate=sample_rate, sentence=None)


def main():
    path = Path('./test.wav')
    recording = load_recording(path)

    a = PronounciationScorer(
        WhisperRecognitionModel.create_from_name("openai/whisper-small.en", num_beams=10),
        None,
        None,
        mistake_finder=MistakeFinderBasic()
    )

    mistakes = a.find_mistakes('Why be conserned with gossip?', recording)

    print(mistakes)


if __name__ == '__main__':
    main()
