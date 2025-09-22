# SpeechCoach

SpeechCoach is an interactive pronunciation helper app designed to help users improve their spoken language skills. The app guides users through a process of practicing sentences, receiving feedback on pronunciation mistakes, and getting personalized suggestions for improvement.

## Quick Introduction

SpeechCoach generates a practice sentence using a Large Language Model (LLM), synthesizes an audio recording of the sentence using a Text-to-Speech (TTS) model, and allows users to record themselves reading the sentence. The app then analyzes the user's recording to detect pronunciation mistakes and provides targeted suggestions for improvement using an LLM.

## How It Works

1. **Sentence Generation**: SpeechCoach uses an LLM (Microsoft Phi-4-mini-instruct) to generate a sentence for the user to practice. This ensures varied and contextually appropriate practice material.

2. **Text-to-Speech (TTS) Synthesis**: The generated sentence is converted into a natural-sounding audio recording using the KokoroTTSModel. This provides users with a reference for correct pronunciation.

3. **User Recording**: Users record themselves reading the sentence directly in the app using a browser-based audio recorder (st_audiorec for Streamlit).

4. **Pronunciation Analysis**: The user's recording is analyzed using the WhisperRecognitionModel (OpenAI Whisper-small.en) to transcribe and score pronunciation. Then the MistakeFinder models identify specific words or sounds that were mispronounced.
   - MistakeFinderCompareWords: Compares the transcribed words with the target sentence to find mismatches.
   - MistakeFinderTokenConfidence: Uses token-level confidence scores from the ASR model to identify low-confidence words that may indicate pronunciation issues.

5. **Mistake Feedback & Suggestions**: Detected mistakes are sent to the LLM, which generates personalized suggestions on how to improve pronunciation for the problematic words or sounds.

## Technologies & Resources Used

- **Streamlit**: For the interactive web frontend, including audio recording and playback.
- **FastAPI**: Backend API for handling requests, running models, and serving results.
- **OpenAI Whisper**: For automatic speech recognition and pronunciation scoring.
- **KokoroTTSModel**: For generating TTS audio from text.
- **Microsoft Phi-4-mini-instruct (LLM)**: For generating practice sentences and personalized feedback.
- **Pydantic**: For data validation and serialization in API requests and responses.

## File Structure Overview

- `backend.py`: FastAPI backend with endpoints for sentence generation, TTS, mistake detection, and suggestions.
- `frontend.py`: Streamlit frontend for user interaction, audio recording, and displaying feedback.
- `src/`: Contains model classes and utility functions:
  - `pronounciation_scorer.py`: Main logic for scoring pronunciation.
  - `mistake_finder.py`: Mistake detection logic.
  - `recognition_model.py`: Whisper ASR integration.
  - `tts_model.py`: TTS model integration.
  - `llm_model.py`: LLM integration for sentence generation and suggestions.
  - `data_classes.py`: Data structures for audio and mistakes.
  - `requests.py`: API request and response pydantic models.
  - `responses.py`: API response pydantic models.
  - `utils.py`: Utility functions for audio processing.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the FastAPI backend**:
   ```bash
   uvicorn backend:app --port 8000 --host <ip_address*>
   ```
   *leave empty if you want to run on localhost
3. **Start the Streamlit frontend**:
   ```bash
   streamlit run frontend.py -- --api_url <url_to_your_backend*>
   ```
   *leave empty if beckend is running on the same machine on port 8000

4. Open the app in your browser and follow the instructions to practice pronunciation!

## License

This project is for educational and research purposes. See individual model licenses for details.

---

SpeechCoach helps you practice, analyze, and improve your pronunciation with the power of modern AI models.
