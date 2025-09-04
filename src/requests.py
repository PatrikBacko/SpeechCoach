from pydantic import BaseModel


class SentenceRequest(BaseModel):
    sentence: str


class FindMistakesRequest(BaseModel):
    sentence: str
    recording_dict: dict


class SuggestionRequest(BaseModel):
    sentence: str
    mistake_dict: dict
