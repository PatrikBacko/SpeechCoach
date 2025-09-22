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


# class MistakeFinderTokenConfidence(MistakeFinderCompareWords):
#     def __init__(
#         self,
#         threshold: float = 0.75
#     ):
#         super().__init__()
#         self._threshold = threshold

#     @override
#     def find_mistakes(
#         self,
#         transcription: Transcription,
#         target_sentence: TargetSentence
#     ) -> list[Mistake]:
#         mistakes = []

#         target_sent_words = self._split_with_indices(target_sentence.sentence)
#         pred_sent_words = self._split_with_indices(transcription.sentence)

#         target_words, target_idxs = zip(*target_sent_words)
#         pred_words, pred_idxs = zip(*pred_sent_words)
#         target_tokens = target_sentence.tokens
#         pred_tokens, pred_tokens_probs = zip(*transcription.tokens_with_probs)

#         _, target_tokens = zip(*self._align_words_with_tokens(target_words, pred_tokens))
#         _, pred_tokens = zip*(self._align_words_with_tokens(pred_words, pred_tokens))

#         for (pred_token, prob), target_token in zip(transcription.tokens_with_probs, target_sentence.tokens):

#             start = transcription.sentence.find(pred_token, mistakes[-1].end_idx + 1 if mistakes else 0)
#             mistakes.append(Mistake(start, start + len(pred_token - 1), None))
#         return mistakes
    
#     def _align_words_with_tokens(
#         words: list[str],
#         tokens: list[str]
#     ) -> list[tuple[str, str]]:
#         aligned = []
#         token_list = []
#         word_idx = 0
#         for token in tokens:
#             if token.startswith(' '):
#                 token_list = []
#                 aligned.append(words[word_idx], token_list)
#                 word_idx += 1
#             token_list.append(token)
#         return aligned


# class MistakeFinderCombined(MistakeFinder):
#     def __init__(
#         self,
#         threshold: float = 0.75
#     ):
#         super().__init__()
#         self._word_finder = MistakeFinderCompareWords()
#         self._token_finder = MistakeFinderTokenConfidence(threshold)

#     @override
#     def find_mistakes(
#         self,
#         transcription: Transcription,
#         target_sentence: TargetSentence
#     ) -> list[Mistake]:
#         word_mistakes = self._word_finder.find_mistakes(transcription, target_sentence)
#         token_mistakes = self._token_finder.find_mistakes(transcription, target_sentence)

#         #TODO        
#         pass


class MistakeFinderType(Enum):
    COMPARE_WORDS = "compare_words"
    # TOKEN_CONFIDENCE = "token_confidence"
    # COMBINED = "combined"


def get_mistake_finder(
    mistake_finder_type: MistakeFinderType,
    **kws
) -> MistakeFinder:
    if mistake_finder_type == MistakeFinderType.COMPARE_WORDS:
        return MistakeFinderCompareWords(**kws)
    elif mistake_finder_type == MistakeFinderType.TOKEN_CONFIDENCE:
        return MistakeFinderTokenConfidence(**kws)
    elif mistake_finder_type == MistakeFinderType.COMBINED:
        return MistakeFinderCombined(**kws)
    else:
        raise ValueError(f"Unknown mistake finder type: {mistake_finder_type}")
