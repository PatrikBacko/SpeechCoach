from abc import ABC, abstractmethod
from typing import override

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.data_classes import AudioRecording, Transcription, TargetSentence


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
    def create_from_name(cls, name: str, num_beams: int = 1) -> "WhisperRecognitionModel":
        processor = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name)
        return cls(processor, model, num_beams)

    @override
    def transcribe(self, recording: AudioRecording) -> Transcription:
        if recording.sample_rate != 16000:
            recording = recording.resample(16000)
        input_features = self._processor(recording.samples, sampling_rate=recording.sample_rate, return_tensors="pt").input_features
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
