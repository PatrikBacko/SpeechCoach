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


class MistakeFinderTokenConfidence(MistakeFinderCompareWords):
    def __init__(
        self,
        threshold: float = 0.8
    ):
        super().__init__()
        self._threshold = threshold

    @override
    def find_mistakes(
        self,
        transcription: Transcription,
        target_sentence: TargetSentence
    ) -> list[Mistake]:
        target_sent_words = self._split_with_indices(target_sentence.sentence)
        pred_sent_words = self._split_with_indices(transcription.sentence)
        pred_words, _ = zip(*pred_sent_words)
        _, pred_tokens = zip(*self._align_words_with_tokens(pred_words, transcription.tokens_with_probs))

        mistakes = []
        for (target_word, target_idx), pred_word, word_tokens in zip(target_sent_words, pred_words, pred_tokens):
            if target_word != pred_word:
                mistakes.append(Mistake(target_idx, target_idx + len(target_word) - 1, pred_word))
            else:
                for token, prob in word_tokens:
                    if token == '.':
                        continue
                    if prob < self._threshold:
                        mistakes.append(Mistake(target_idx, target_idx + len(target_word) - 1, None))
                        break
        return mistakes


    def _align_words_with_tokens(
        self,
        words: list[str],
        tokens_with_probs: list[tuple[str, int]]
    ) -> list[tuple[str, tuple[str, int]]]:
        aligned = []
        token_list = []
        word_idx = 0
        for token, prob in tokens_with_probs:
            if token.startswith(' '):
                token_list = []
                aligned.append((words[word_idx], token_list))
                word_idx += 1
            token_list.append((token, prob))
        return aligned


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
