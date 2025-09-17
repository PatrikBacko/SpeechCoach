from abc import ABC, abstractmethod
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


class MistakeFinderBasic(MistakeFinder):
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
