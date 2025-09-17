from fastapi import FastAPI

# from src.pronounciation_scorer import PronounciationScorer
# from src.mistake_finder import MistakeFinderBasic
# from src.recognition_model import WhisperRecognitionModel
# from src.tts_model import KokoroTTSModel
# from src.llm_model import HuggingFaceLLMModel, TargetSentencePrompt, SuggestionPrompt
from src.data_classes import AudioRecording, Mistake
from src.requests import SentenceRequest, FindMistakesRequest, SuggestionRequest
from src.responses import TargetSentenceResponse, AudioRecordingResponse, MistakeResponse, SuggestionResponse


app = FastAPI()


# scorer = PronounciationScorer(
#     WhisperRecognitionModel.create_from_name("openai/whisper-small.en", num_beams=10),
#     KokoroTTSModel(),
#     HuggingFaceLLMModel.create_from_name(
#         "microsoft/Phi-4-mini-instruct",
#         TargetSentencePrompt(),
#         SuggestionPrompt(),
#     ),
#     mistake_finder=MistakeFinderBasic()
# )


@app.post("/generate_target_sentence", response_model=TargetSentenceResponse)
def generate_target_sentence() -> TargetSentenceResponse:
    sentence = "She sells seashells by the seashore."
    return TargetSentenceResponse(target_sentence=sentence)


@app.post("/generate_tts", response_model=AudioRecordingResponse)
def generate_tts(request: SentenceRequest) -> AudioRecordingResponse:
    print(request)
    path = '/home/buciak/Desktop/mff/24-25/summer_semester/speech_recognition/test.wav'
    with open(path, "rb") as f:
        wav_bytes = f.read()
    audio_recording = AudioRecording.from_wav_bytes(wav_bytes, sentence=request.sentence)
    return AudioRecordingResponse(audio_data=audio_recording.to_json())


@app.post("/find_mistakes", response_model=MistakeResponse)
def find_mistakes(request: FindMistakesRequest) -> MistakeResponse:
    print(request.sentence)
    mistakes = [
        Mistake(4, 8, 'selz'),
        Mistake(20, 21, 'baj')
    ]
    return MistakeResponse(mistakes=[mistake.to_json() for mistake in mistakes])


@app.post("/generate_suggestion", response_model=SuggestionResponse)
def generate_suggestion(request: SuggestionRequest) -> SuggestionResponse:
    mistake = Mistake.from_json(request.mistake_dict)
    suggestion = f'{mistake}: you should kill yourself.'
    return SuggestionResponse(suggestion=suggestion)
