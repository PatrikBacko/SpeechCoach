from abc import ABC, abstractmethod
from enum import Enum
import re
from typing import override

from src.data_classes import Transcription, Mistake, TargetSentence


class MistakeFinder(ABC):
    @abstractmethod
    def find_mistakes(
        self,
        transcription: Transcription,
        target_sentence: TargetSentence,
    ) -> list[Mistake]:
        pass


class MistakeFinderCompareWords(MistakeFinder):
    def __init__(self):
        super().__init__()

    @override
    def find_mistakes(
        self,
        transcription: Transcription,
        target_sentence: TargetSentence
    ) -> list[Mistake]:
        target_sent_words = self._split_with_indices(target_sentence.sentence)
        pred_sent_words = self._split_with_indices(transcription.sentence)
        mistakes = [
            Mistake(target_idx, target_idx + len(target_word) - 1, pred_word)
            for (target_word, target_idx), (pred_word, _) in zip(target_sent_words, pred_sent_words)
            if target_word != pred_word
        ]
        return mistakes

    def _split_with_indices(
        self,
        text: str,
        delimiter: str = ' '
    ) -> list[tuple[str, int]]:
        pattern = re.compile(r'[^' + re.escape(delimiter) + r']+')
        return [(m.group(), m.start()) for m in pattern.finditer(text)]


class MistakeFinderTokenConfidence(MistakeFinder):
    def __init__(
        self,
        threshold: float = 0.75
    ):
        super().__init__()
        self._threshold = threshold

    @override
    def find_mistakes(
        self,
        transcription: Transcription,
        target_sentence: TargetSentence
    ) -> list[Mistake]:
        mistakes = []
        #TODO the tokens are not alligned properly, probably do the words comaprison first and then check whether the tokens in the word are below the threshold
        for (pred_token, prob), target_token in zip(transcription.tokens_with_probs, target_sentence.tokens):
            if pred_token != target_token:
                start = target_sentence.sentence.find(target_token)
                mistakes.append(Mistake(start, start + len(pred_token) - 1, pred_token))
            elif prob < self._threshold:
                start = transcription.sentence.find(pred_token, mistakes[-1].end_idx + 1 if mistakes else 0)
                mistakes.append(Mistake(start, start + len(pred_token - 1), None))
        return mistakes


class MistakeFinderType(Enum):
    COMPARE_WORDS = "compare_words"
    TOKEN_CONFIDENCE = "token_confidence"


def get_mistake_finder(
    mistake_finder_type: MistakeFinderType,
    **kws
) -> MistakeFinder:
    if mistake_finder_type == MistakeFinderType.COMPARE_WORDS:
        return MistakeFinderCompareWords(**kws)
    elif mistake_finder_type == MistakeFinderType.TOKEN_CONFIDENCE:
        return MistakeFinderTokenConfidence(**kws)
    else:
        raise ValueError(f"Unknown mistake finder type: {mistake_finder_type}")
