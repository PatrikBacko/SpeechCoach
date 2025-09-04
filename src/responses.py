from pydantic import BaseModel
from typing import Any


class TargetSentenceResponse(BaseModel):
    target_sentence: str


class AudioRecordingResponse(BaseModel):
    audio_data: dict[str, Any]


class MistakeResponse(BaseModel):
    mistakes: list[dict[str, Any]]


class SuggestionResponse(BaseModel):
    suggestion: str
