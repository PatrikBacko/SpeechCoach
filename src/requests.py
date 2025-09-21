from pydantic import BaseModel
from src.mistake_finder import MistakeFinderType

class SentenceRequest(BaseModel):
    sentence: str


class FindMistakesRequest(BaseModel):
    sentence: str
    recording_dict: dict


class SuggestionRequest(BaseModel):
    sentence: str
    mistake_dict: dict

class ChangeMistakeFinderRequest(BaseModel):
    type: MistakeFinderType
