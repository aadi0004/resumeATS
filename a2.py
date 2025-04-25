import os
import queue
import tempfile
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
import whisper
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# ─── SETUP ─────────────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
st.set_page_config(page_title="Voice HR Interview", layout="centered")
st.title("🎤 Voice-based HR Interview (No TTS)")

# ─── SESSION VARS ──────────────────────────────────────────────────────────────
if "questions" not in st.session_state:
    st.session_state.questions = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# ─── UPLOAD RESUME ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
if uploaded_file:
    resume_text = ""
    for page in PdfReader(uploaded_file).pages:
        resume_text += page.extract_text() or ""
    st.session_state.resume_text = resume_text
    st.success("✅ Resume uploaded successfully.")

# ─── GEMINI Qs & FEEDBACK ──────────────────────────────────────────────────────
@st.cache_data
def generate_questions(resume):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"You're an HR interviewer. Generate 3 short interview questions based on this resume:\n{resume}"
    response = model.generate_content(prompt)
    return [line.strip() for line in response.text.split("\n") if line.strip()]

def get_feedback(question, answer):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"The candidate was asked: '{question}'. Their response was: '{answer}'. Give a score out of 10 and a brief constructive feedback."
    response = model.generate_content(prompt)
    return response.text

# ─── GENERATE QUESTIONS ────────────────────────────────────────────────────────
if st.session_state.resume_text and st.button("Generate Interview Questions"):
    st.session_state.questions = generate_questions(st.session_state.resume_text)

# ─── AUDIO CAPTURE SETUP ───────────────────────────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(frame):
    audio = frame.to_ndarray().flatten().tobytes()
    audio_queue.put(audio)
    return frame

client_settings = ClientSettings(media_stream_constraints={"audio": True, "video": False})

# ─── QA LOOP ───────────────────────────────────────────────────────────────────
if st.session_state.questions:
    for idx, question in enumerate(st.session_state.questions):
        st.subheader(f"🧠 Question {idx+1}")
        st.write(question)

        ctx = webrtc_streamer(
            key=f"audio_{idx}",
            mode=WebRtcMode.SENDRECV,
            client_settings=client_settings,
            audio_frame_callback=audio_callback,
        )

        if st.button(f"Submit Answer {idx+1}"):
            if audio_queue.empty():
                st.warning("⚠️ No audio captured. Please try again.")
            else:
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                with open(temp_audio.name, "wb") as f:
                    while not audio_queue.empty():
                        f.write(audio_queue.get())

                with st.spinner("Transcribing your answer..."):
                    try:
                        model = whisper.load_model("base")
                        result = model.transcribe(temp_audio.name)
                        answer = result["text"]
                        st.success("📝 Your Answer:")
                        st.write(answer)

                        st.markdown("🔍 **Feedback from HR AI:**")
                        feedback = get_feedback(question, answer)
                        st.write(feedback)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
else:
    st.info("📄 Upload your resume and click 'Generate Interview Questions' to begin.")
