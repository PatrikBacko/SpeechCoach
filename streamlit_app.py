import streamlit as st
from st_audiorec import st_audiorec
import requests
from pydantic import BaseModel
from typing import List, Dict

from src.data_classes import AudioRecording


API_URL = "http://n21:8000/"


class MistakeResponse(BaseModel):
    mistakes: List[Dict]

class SuggestionResponse(BaseModel):
    suggestion: str


st.title("Pronunciation Scorer App")


gen_sentence = st.button("Generate Target Sentence")
if gen_sentence:
    resp = requests.post(f"{API_URL}/generate_target_sentence")
    if resp.ok:
        st.session_state["target_sentence"] = resp.json()["target_sentence"]
    else:
        st.error("Failed to generate target sentence.")


sentence = st.text_input("Sentence to practice", value=st.session_state.get("target_sentence", ""))


gen_tts = st.button("Generate TTS Audio")
if gen_tts and sentence:
    resp = requests.post(f"{API_URL}/generate_tts", json={"sentence": sentence})
    if resp.ok:
        audio_recording = AudioRecording.from_json(resp.json())
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
        data = {
            'sentence': sentence,
            'recording_dict': audio_recording.to_json(),
        }
        resp = requests.post(f"{API_URL}/find_mistakes", json=data)
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
        data = {"sentence": sentence, "mistake_dict": st.session_state["mistake"]}
        resp = requests.post(f"{API_URL}/generate_suggestion", json=data)
        if resp.ok:
            suggestion_response = SuggestionResponse(**resp.json())
            st.success(f"Suggestion: {suggestion_response.suggestion}")
        else:
            st.error("Failed to generate suggestion.")
