import argparse
import requests

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from src.data_classes import AudioRecording, Mistake
from src.requests import SentenceRequest, FindMistakesRequest, SuggestionRequest, ChangeMistakeFinderRequest
from src.mistake_finder import MistakeFinderType
from src.responses import TargetSentenceResponse, AudioRecordingResponse, MistakeResponse, SuggestionResponse


def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000",
        help="URL of the backend API"
    )
    return parser


def highlight_text_with_mistakes(text: str, mistakes: list[Mistake], size: float = 1.5) -> str:
    """
    Highlight parts of text in red with tooltips on hover, and make text bigger.

    :param text: The full text string.
    :param mistakes: List of Mistake objects.
    :param size: Font size in em units (default 1.5).
    :return: HTML formatted string.
    """
    result = f'<span style="font-size:{size}em;">'
    last_idx = 0

    for mistake in sorted(mistakes, key=lambda x: x.start_idx):
        start, end = mistake.start_idx, mistake.end_idx + 1

        # green text before red span
        result += f'<span style="color:green">{text[last_idx:start]}</span>'

        # red text with tooltip
        result += (
            f'<span style="color:red">'
            f'{text[start:end]}'
            f'</span>'
        )

        last_idx = end

    # remaining green text after last span
    result += f'<span style="color:green">{text[last_idx:]}</span>'
    result += '</span>'

    return result

parser = return_parser()
args = parser.parse_args()
API_URL = args.api_url

st.title("SpeechCoach")

gen_sentence = st.button("Generate Target Sentence")
if gen_sentence:
    with st.spinner("Generating target sentence..."):
        resp = requests.post(f"{API_URL}/generate_target_sentence")
    if resp.ok:
        st.session_state["target_sentence"] = TargetSentenceResponse(**resp.json()).target_sentence
    else:
        st.error("Failed to generate target sentence.")

sentence = st.text_input("Sentence to practice", value=st.session_state.get("target_sentence", ""))

gen_tts = st.button("Generate TTS Audio")
if gen_tts and sentence:
    with st.spinner("Generating TTS audio..."):
        request = SentenceRequest(sentence=sentence)
        resp = requests.post(f"{API_URL}/generate_tts", data=request.model_dump_json())
    if resp.ok:
        audio_recording = AudioRecording.from_json(AudioRecordingResponse(**resp.json()).audio_data)
        st.audio(audio_recording.to_wav_bytes(), format="audio/wav")
    else:
        st.error("Failed to generate TTS audio.")

st.header("Record Your Pronunciation")
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    key="speech",
    format='wav',
)
audio_file = None
if audio and "bytes" in audio:
    st.audio(audio["bytes"], format="audio/wav")
    audio_file = ("user_recording.wav", audio["bytes"], "audio/wav")

st.header('Evaluate Your Pronunciation')

#select mistake finder from options
mistake_finder_type = st.selectbox(
    'Select Mistake Finder Type',
    (tp.value for tp in MistakeFinderType),
    index=0
)
if st.button("Change Mistake Finder"):
    request = ChangeMistakeFinderRequest(type=MistakeFinderType(mistake_finder_type))
    resp = requests.post(f"{API_URL}/change_mistake_finder", data=request.model_dump_json())
    if resp.ok:
        st.success(f"Mistake finder changed to {mistake_finder_type}.")
    else:
        st.error("Failed to change mistake finder.")

if st.button("Find Mistakes"):
    if sentence and audio_file:
        # Convert WebM bytes to WAV bytes
        audio_recording = AudioRecording.from_wav_bytes(audio["bytes"])
        with st.spinner("Finding mistakes..."):
            request = FindMistakesRequest(**{
                'sentence': sentence,
                'recording_dict': audio_recording.to_json(),
            })
            resp = requests.post(f"{API_URL}/find_mistakes", data=request.model_dump_json())
        if resp.ok:
            mistake_response = MistakeResponse(**resp.json())
            mistakes = [Mistake.from_json(mistake_data) for mistake_data in mistake_response.mistakes]
            st.markdown(highlight_text_with_mistakes(sentence, mistakes, size=2), unsafe_allow_html=True)
            for idx, mistake in enumerate(mistakes):
                st.subheader(
                    sentence[mistake.start_idx:mistake.end_idx+1] +
                    (f' -> {mistake.mistaken_text}' if mistake.mistaken_text is not None else '')
                )
                with st.spinner("Generating suggestion..."):
                    request = SuggestionRequest(sentence=sentence, mistake_dict=mistake.to_json())
                    resp = requests.post(f"{API_URL}/generate_suggestion", data=request.model_dump_json())
                if resp.ok:
                    suggestion_response = SuggestionResponse(**resp.json())
                    st.success(f"Suggestion:\n{suggestion_response.suggestion}")
                else:
                    st.error("Failed to generate suggestion for this mistake.")
        else:
            st.error("Failed to find mistakes.")
    else:
        st.error("Please first generate a sentence and record it.")
