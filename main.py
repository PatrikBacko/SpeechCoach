from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any
from src.pronounciation_scorer import PronounciationScorer
from src.mistake_finder import MistakeFinderBasic
from src.recognition_model import WhisperRecognitionModel
from src.tts_model import KokoroTTSModel
from src.llm_model import HuggingFaceLLMModel, TargetSentencePrompt, SuggestionPrompt
from src.data_classes import AudioRecording, Mistake


app = FastAPI()


scorer = PronounciationScorer(
    WhisperRecognitionModel.create_from_name("openai/whisper-small.en", num_beams=10),
    KokoroTTSModel(),
    HuggingFaceLLMModel.create_from_name(
        "microsoft/Phi-4-mini-instruct",
        TargetSentencePrompt(),
        SuggestionPrompt(),
    ),
    mistake_finder=MistakeFinderBasic()
)


class SentenceRequest(BaseModel):
    sentence: str

class FindMistakesRequest(BaseModel):
    sentence: str
    recording_dict: dict

class SuggestionRequest(BaseModel):
    sentence: str
    mistake_dict: dict

class TargetSentenceResponse(BaseModel):
    target_sentence: str

class AudioRecordingResponse(BaseModel):
    audio_data: Dict[str, Any]

class MistakeResponse(BaseModel):
    mistakes: List[Dict[str, Any]]

class SuggestionResponse(BaseModel):
    suggestion: str


@app.post("/generate_target_sentence", response_model=TargetSentenceResponse)
def generate_target_sentence() -> TargetSentenceResponse:
    sentence = scorer.generate_target_sentence()
    return TargetSentenceResponse(target_sentence=sentence)


@app.post("/generate_tts", response_model=AudioRecordingResponse)
def generate_tts(request: SentenceRequest) -> AudioRecordingResponse:
    audio_recording = scorer.generate_tts(request.sentence)
    return AudioRecordingResponse(audio_data=audio_recording.to_json())


@app.post("/find_mistakes", response_model=MistakeResponse)
def find_mistakes(request: FindMistakesRequest) -> MistakeResponse:
    recording = AudioRecording.from_json(request.recording_dict)
    mistakes = scorer.find_mistakes(request.sentence, recording)
    return MistakeResponse(mistakes=[mistake.to_json() for mistake in mistakes])


@app.post("/generate_suggestion", response_model=SuggestionResponse)
def generate_suggestion(request: SuggestionRequest) -> SuggestionResponse:
    mistake = Mistake.from_json(request.mistake_dict)
    suggestion = scorer.generetare_suggestion(mistake, request.sentence)
    return SuggestionResponse(suggestion=suggestion)
