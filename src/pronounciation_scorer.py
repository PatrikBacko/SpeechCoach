from src.data_classes import AudioRecording, Mistake
from src.tts_model import TTSModel
from src.recognition_model import RecognitionModel
from src.llm_model import LLMModel
from src.mistake_finder import MistakeFinder

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
        target_sentence = self._recognition_model.tokenize_sentence(sentence)
        return self._mistake_finder.find_mistakes(transcription, target_sentence)
    
    def generetare_suggestion(
        self,
        mistake: Mistake,
        sentence: str,
    ) -> str:
        return self._llm_model.generate_suggestions(mistake, sentence)

    def generate_target_sentence(
        self
    ) -> str:
        return self._llm_model.generate_target_sentence()
    
    def generate_tts(
        self,
        sentence: str
    ) -> AudioRecording:
        return self._tts_model.synthesize(sentence)

    def synthesize(
        self,
        sentence: str
    ) -> AudioRecording:
        return self._tts_model.synthesize(sentence)

    def change_mistake_finder(
        self,
        mistake_finder: MistakeFinder
    ) -> None:
        self._mistake_finder = mistake_finder
