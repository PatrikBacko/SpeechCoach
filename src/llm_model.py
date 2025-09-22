from abc import ABC, abstractmethod
from typing import override

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.data_classes import Mistake
from src.utils import get_auto_device


type Prompt = list[dict[str, str]]


class TargetSentencePrompt:
    def __call__(
        self
    ):
        return [
            {"role": "system", "content": "You are a worker in an English pronunciation education application. "
                                        "Generate only one medium-sized sentence in English with arbitrary topic. "
                                        "Do not include explanations, instructions, or anything else. Output only the sentence."},
]


class SuggestionPrompt:
    def __call__(
        self,
        sentence: str,
        mistake_word: str,
        incorrect_pronunciation: str | None = None,
    ):
        return [
            {"role": "system", "content": "You are a helpful pronunciation assistant. " 
                                        "Your task is to help users improve their pronunciation by: "
                                        + ("- Identifying the difference between the correct word and what the user said incorrectly. "
                                            if incorrect_pronunciation is not None
                                            else ''
                                        ) +
                                        "- Explaining how the correct word should be pronounced, using simple phonetic hints or IPA (International Phonetic Alphabet). "
                                        "- Giving a short, clear tip on how to shape the mouth, tongue, or voice to pronounce correctly. "
                                        "- Keeping explanations concise, supportive, and easy to follow. "
                                        "Always respond in a friendly, encouraging way. "
                                        "Do not overwhelm the user with long linguistic theory — focus on practical help. "
                                        f"Sentence: {sentence} "
                                        f"Word where mispronouciation was detected: {mistake_word}"
                                        + (f"-> {incorrect_pronunciation} (the mispronunciation detected by the system)."
                                            if incorrect_pronunciation is not None
                                            else ''
                                        ) +
                                        f"Please help to the user: "
            },
        ]


class LLMModel(ABC):
    @abstractmethod
    def generate_target_sentence(self) -> str:
        ...

    @abstractmethod
    def generate_suggestions(self, mistake: Mistake, sentence: str) -> str:
        ...


class HuggingFaceLLMModel(LLMModel):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        target_sentence_prompt: TargetSentencePrompt,
        suggestion_prompt: SuggestionPrompt,
        target_sentence_temperature: float = 1.0,
        target_sentence_do_sample: bool = True,
        suggestion_temperature: float = 0.0,
        suggestion_do_sample: bool = False,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._target_sentence_prompt = target_sentence_prompt
        self._suggesstion_prompt = suggestion_prompt
        self._target_sentence_temperature = target_sentence_temperature
        self._target_sentence_do_sample = target_sentence_do_sample
        self._suggestion_temperature = suggestion_temperature
        self._suggestion_do_sample = suggestion_do_sample
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
        )

    @classmethod
    def create_from_name(
        cls,
        name: str,
        target_sentence_prompt: str,
        suggestion_prompt: str,
        device: torch.device | str | None = None
    ) -> 'HuggingFaceLLMModel':
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map=device or get_auto_device(),
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(name)
        return cls(model, tokenizer, target_sentence_prompt, suggestion_prompt)

    @override
    def generate_target_sentence(self) -> str:
        messages = self._target_sentence_prompt()
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": self._target_sentence_temperature,
            "do_sample": self._target_sentence_do_sample,
        }
        output = self._pipe(messages, **generation_args)
        return output[0]['generated_text']

    @override
    def generate_suggestions(self, mistake: Mistake, sentence: str) -> str:
        messages = self._suggesstion_prompt(
            sentence,
            mistake.mistaken_text,
            sentence[mistake.start_idx:mistake.end_idx]
        )
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": self._suggestion_temperature,
            "do_sample": self._suggestion_do_sample,
        }
        output = self._pipe(messages, **generation_args)
        return output[0]['generated_text']
