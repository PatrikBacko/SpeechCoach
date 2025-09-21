from pathlib import Path

from src.pronounciation_scorer import PronounciationScorer
from src.mistake_finder import MistakeFinderCompareWords
from src.recognition_model import WhisperRecognitionModel
from src.tts_model import KokoroTTSModel
from src.llm_model import HuggingFaceLLMModel
from src.llm_model import TargetSentencePrompt, SuggestionPrompt

import warnings
warnings.filterwarnings("ignore")

def main():
    a = PronounciationScorer(
        WhisperRecognitionModel.create_from_name("openai/whisper-small.en", num_beams=10),
        KokoroTTSModel(),
        HuggingFaceLLMModel.create_from_name(
            "microsoft/Phi-4-mini-instruct",
            TargetSentencePrompt(),
            SuggestionPrompt(),
        ),
        mistake_finder=MistakeFinderCompareWords()
    )

    target_sentence = a.generate_target_sentence()
    print(f'Target sentence: {target_sentence}')
    tts_recording = a.generate_tts(target_sentence)
    mistakes = a.find_mistakes(target_sentence, tts_recording)
    print(f'Mistakes: {mistakes}')
    for mistake in mistakes:
        suggestion = a.generetare_suggestion(mistake, target_sentence)
        print(f'Suggestion: {suggestion}')


if __name__ == '__main__':
    main()
