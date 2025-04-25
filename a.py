import os
import requests
import queue
import tempfile
import wave
import numpy as np

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
import whisper
import imageio_ffmpeg
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ─── FFMPEG SETUP ─────────────────────────────────────────────────────────────
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

if not GEMINI_API_KEY or not ELEVEN_API_KEY:
    st.error("Please set both GOOGLE_API_KEY and ELEVEN_API_KEY in your .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ─── UI SETUP ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Voice HR Interview", layout="centered")
st.title("🎤 Voice-based HR Interview with AI")

# ─── RESUME UPLOAD ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
if uploaded_file:
    text = ""
    for p in PdfReader(uploaded_file).pages:
        text += p.extract_text() or ""
    st.session_state["resume_text"] = text
    st.success("✅ Resume uploaded.")

# ─── GEMINI Qs & FEEDBACK ──────────────────────────────────────────────────────
@st.cache_data
def generate_questions(resume):
    m = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"You're an HR interviewer. Generate 1 concise interview questions small based on this resume:\n{resume}"
    return [q.strip() for q in m.generate_content(prompt).text.splitlines() if q.strip()]

def get_feedback(question, answer):
    m = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"The candidate was asked: '{question}'. They answered: '{answer}'. Provide a score out of 10 and constructive feedback."
    return m.generate_content(prompt).text

# ─── ELEVENLABS VOICES ────────────────────────────────────────────────────────
@st.cache_data
def fetch_voices():
    resp = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": ELEVEN_API_KEY})
    return {v["name"]: v["voice_id"] for v in resp.json().get("voices", [])}

voices = fetch_voices()
voice = st.selectbox("Select AI Interviewer Voice", list(voices), index=0)
voice_id = voices[voice]

def speak(text, idx):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_22050_32"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": "eleven_monolingual_v1", "voice_settings": {"stability":0.7,"similarity_boost":0.8}}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        path = f"q{idx}.mp3"
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    else:
        st.error(f"TTS Error {r.status_code}: {r.text}")
        return None

# ─── AUDIO QUEUES ──────────────────────────────────────────────────────────────
audio_queues = {}

def get_audio_callback(qkey):
    def callback(frame):
        audio = frame.to_ndarray()
        if qkey not in audio_queues:
            audio_queues[qkey] = []
        audio_queues[qkey].append(audio)
        return frame
    return callback

# ─── GENERATE QUESTIONS ────────────────────────────────────────────────────────
if "resume_text" in st.session_state and st.button("Generate Interview Questions"):
    st.session_state["questions"] = generate_questions(st.session_state["resume_text"])

# ─── QA LOOP ───────────────────────────────────────────────────────────────────
if "questions" in st.session_state:
    for i, q in enumerate(st.session_state["questions"], 1):
        st.subheader(f"Question {i}")
        st.write(q)

        mp3 = speak(q, i)
        if mp3:
            st.audio(mp3)

        qkey = f"audio_{i}"
        st.markdown("**🎙 Record your answer below:**")
        ctx = webrtc_streamer(
            key=qkey,
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            sendback_audio=False,
            audio_frame_callback=get_audio_callback(qkey)
        )

        if st.button(f"Submit Answer {i}"):
            if qkey not in audio_queues or len(audio_queues[qkey]) == 0:
                st.warning("No audio captured; please speak into your mic.")
            else:
                tmp = tempfile.mktemp(".wav")

                # Save to WAV
                with wave.open(tmp, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(16000)
                    audio_data = np.concatenate(audio_queues[qkey])
                    wf.writeframes(audio_data.astype(np.int16).tobytes())

                with st.spinner("Transcribing…"):
                    try:
                        model = whisper.load_model("base")
                        res = model.transcribe(tmp)
                        ans = res.get("text", "")
                        st.markdown(f"📝 **Your Answer:** {ans}")
                        fb = get_feedback(q, ans)
                        st.markdown("📋 **Feedback:**")
                        st.write(fb)
                        st.audio(tmp)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
else:
    st.info("Upload your resume and click ‘Generate Interview Questions’ to begin.")
