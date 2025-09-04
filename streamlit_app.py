import streamlit as st
from st_audiorec import st_audiorec
import requests

from src.data_classes import AudioRecording
from src.requests import SentenceRequest, FindMistakesRequest, SuggestionRequest
from src.responses import TargetSentenceResponse, AudioRecordingResponse, MistakeResponse, SuggestionResponse


API_URL = "http://n25:8000/"


st.title("Pronunciation Scorer App")


gen_sentence = st.button("Generate Target Sentence")
if gen_sentence:
    resp = requests.post(f"{API_URL}/generate_target_sentence")
    if resp.ok:
        st.session_state["target_sentence"] = TargetSentenceResponse(**resp.json()).target_sentence
    else:
        st.error("Failed to generate target sentence.")


sentence = st.text_input("Sentence to practice", value=st.session_state.get("target_sentence", ""))


gen_tts = st.button("Generate TTS Audio")
if gen_tts and sentence:
    request = SentenceRequest(sentence=sentence)
    resp = requests.post(f"{API_URL}/generate_tts", data=request)
    if resp.ok:
        audio_recording = AudioRecording.from_json(AudioRecordingResponse(**resp.json()).audio_data)
        st.audio(audio_recording.to_wav_bytes(), format="audio/wav")
    else:
        st.error("Failed to generate TTS audio.")


st.header("Record Your Pronunciation")
audio_bytes = st_audiorec()
audio_file = None
if audio_bytes:
    audio_file = ("user_recording.wav", audio_bytes, "audio/wav")


if st.button("Find Mistakes"):
    if sentence and audio_file:
        audio_recording = AudioRecording.from_wav_bytes(audio_bytes)
        request = FindMistakesRequest(**{
            'sentence': sentence,
            'recording_dict': audio_recording.to_json(),
        })
        resp = requests.post(f"{API_URL}/find_mistakes", data=request)
        if resp.ok:
            mistake_response = MistakeResponse(**resp.json())
            st.write("Mistakes:", mistake_response.mistakes)
            if mistake_response.mistakes:
                st.session_state["mistake"] = mistake_response.mistakes[0]
        else:
            st.error("Failed to find mistakes.")
    else:
        st.error("Please first generate a sentence and record it.")


if "mistake" in st.session_state:
    if st.button("Generate Suggestion for Mistake"):
        request = SuggestionRequest(**{"sentence": sentence, "mistake_dict": st.session_state["mistake"]})
        resp = requests.post(f"{API_URL}/generate_suggestion", data=request)
        if resp.ok:
            suggestion_response = SuggestionResponse(**resp.json()).suggestion
            st.success(f"Suggestion: {suggestion_response.suggestion}")
        else:
            st.error("Failed to generate suggestion.")
