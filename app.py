import logging
import io
import os
import json
import threading
import csv
import random
import string
from datetime import datetime, timedelta
from typing import Optional, Union
import pandas as pd
import streamlit as st
from PIL import Image
import pdf2image
import google.generativeai as genai
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import streamlit.components.v1 as components
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        st.error("Google API configuration failed. Please check your GOOGLE_API_KEY.")
else:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")

# Initialize ElevenLabs client
try:
    client = ElevenLabs(api_key=ELEVEN_API_KEY)
    logger.info("ElevenLabs client initialized")
except Exception as e:
    logger.error(f"Failed to initialize ElevenLabs client: {e}")
    st.error("ElevenLabs initialization failed. Please check your ELEVEN_API_KEY.")

# Initialize database connection pool
try:
    DB_POOL = SimpleConnectionPool(
        1, 10,
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        dbname=os.getenv("PG_DB")
    )
    logger.info("Database connection pool initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database pool: {e}")
    st.error(f"Database connection failed: {e}")
    raise

# -------------------- LOGGING SETUP --------------------
csv_lock = threading.Lock()
LOG_DIR = ".logs"
LOG_FILE = os.path.join(LOG_DIR, "api_usage_logs.csv")

os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Action", "API_Hits", "Tokens_Generated", "Total_Tokens_Till_Now"])

def get_current_total_tokens():
    total = 0
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    total += int(row.get("Tokens_Generated", 0))
                except ValueError:
                    continue
    return total

def log_api_usage(action: str, tokens_generated: int):
    with csv_lock:
        current_total = get_current_total_tokens()
        new_total = current_total + tokens_generated
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    action,
                    1,
                    tokens_generated,
                    new_total
                ])
            logger.info(f"Logged API usage: {action}, tokens: {tokens_generated}")
        except Exception as e:
            logger.error(f"Failed to log API usage: {e}")

# -------------------- DATABASE FUNCTIONS --------------------
def init_db():
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255),
                mobile_number VARCHAR(20),
                theme VARCHAR(10) DEFAULT 'Light',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS button_logs (
                id SERIAL PRIMARY KEY,
                action VARCHAR(255),
                response TEXT,
                user_id INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255),
                resume_text TEXT,
                user_id INTEGER REFERENCES users(id),
                version_label VARCHAR(255),
                version_number INTEGER DEFAULT 1,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_applications (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                company_name VARCHAR(255),
                job_role VARCHAR(255),
                application_date DATE,
                resume_id INTEGER REFERENCES resumes(id),
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                id SERIAL PRIMARY KEY,
                action VARCHAR(255),
                prompt TEXT,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                action VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_goals (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                goal TEXT,
                status VARCHAR(50) DEFAULT 'Pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_alerts (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                job_title VARCHAR(255),
                company VARCHAR(255),
                description TEXT,
                apply_link VARCHAR(512),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        st.error(f"Database initialization failed: {e}")
    finally:
        DB_POOL.putconn(conn)

def register_user(username: str, email: str, mobile_number: str) -> bool:
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, mobile_number) VALUES (%s, %s, %s)", (username, email, mobile_number))
        conn.commit()
        cursor.close()
        logger.info(f"User registered: {username}")
        return True
    except psycopg2.IntegrityError:
        st.error("Username already exists.")
        logger.warning(f"Registration failed for {username}: Username exists")
        return False
    except Exception as e:
        st.error(f"Registration failed: {e}")
        logger.error(f"Registration failed for {username}: {e}")
        return False
    finally:
        DB_POOL.putconn(conn)

def login_user(username: str) -> bool:
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        if result:
            st.session_state.username = username
            logger.info(f"User logged in: {username}")
            return True
        else:
            st.error("Username not found.")
            logger.warning(f"Login failed for {username}: Not found")
            return False
    except Exception as e:
        st.error(f"Login failed: {e}")
        logger.error(f"Login failed for {username}: {e}")
        return False
    finally:
        DB_POOL.putconn(conn)

def log_to_postgres(action: str, response: str):
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
        user_id = cursor.fetchone()
        if user_id:
            cursor.execute("INSERT INTO button_logs (action, response, user_id) VALUES (%s, %s, %s)", (action, response, user_id[0]))
            conn.commit()
            logger.info(f"Logged action to Postgres: {action}")
        cursor.close()
    except Exception as e:
        st.error(f"PostgreSQL logging failed: {e}")
        logger.error(f"PostgreSQL logging failed: {e}")
    finally:
        DB_POOL.putconn(conn)

def save_resume_to_postgres(filename: str, resume_text: str, version_label: str, version_number: int = 1) -> Union[int, None]:
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
        user_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT id, version_number FROM resumes WHERE filename = %s AND user_id = %s ORDER BY version_number DESC LIMIT 1",
            (filename, user_id)
        )
        existing_resume = cursor.fetchone()
        if existing_resume:
            new_version = existing_resume[1] + 1
            cursor.execute(
                "INSERT INTO resumes (filename, resume_text, user_id, version_label, version_number) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (filename, resume_text, user_id, version_label, new_version)
            )
        else:
            cursor.execute(
                "INSERT INTO resumes (filename, resume_text, user_id, version_label, version_number) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (filename, resume_text, user_id, version_label, version_number)
            )
        resume_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        logger.info(f"Saved resume: {filename}, version: {version_label}, version_number: {version_number}")
        return resume_id
    except Exception as e:
        logger.error(f"Failed to save resume: {e}")
        return None
    finally:
        DB_POOL.putconn(conn)

def check_api_quota(user_id: int) -> bool:
    DAILY_QUOTA = 50
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM api_usage WHERE user_id = %s AND timestamp > %s",
            (user_id, datetime.now() - timedelta(days=1))
        )
        count = cursor.fetchone()[0]
        cursor.close()
        if count >= DAILY_QUOTA:
            st.error("Daily API quota reached. Try again tomorrow.")
            logger.warning(f"User {user_id} reached API quota")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to check API quota: {e}")
        return False
    finally:
        DB_POOL.putconn(conn)

def log_api_call(user_id: int, action: str):
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO api_usage (user_id, action) VALUES (%s, %s)", (user_id, action))
        conn.commit()
        cursor.close()
    except Exception as e:
        logger.error(f"Failed to log API call: {e}")
    finally:
        DB_POOL.putconn(conn)

# -------------------- Gemini API Wrapper --------------------
def get_gemini_response(prompt: str, action: str = "Gemini_API_Call") -> str:
    if not prompt.strip():
        logger.warning("Empty prompt provided to for Gemini API")
        return "Error: Prompt is empty. Please provide a valid prompt."
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
        user_id = cursor.fetchone()[0]
        if not check_api_quota(user_id):
            cursor.close()
            DB_POOL.putconn(conn)
            return "Error: API quota exceeded."
        cursor.execute("SELECT response FROM api_cache WHERE action = %s AND prompt = %s", (action, prompt))
        cached = cursor.fetchone()
        if cached:
            cursor.close()
            DB_POOL.putconn(conn)
            logger.info(f"Retrieved cached Gemini response for action: {action}")
            return cached[0]
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, f"Add randomness: {os.urandom(8).hex()}"])
        if hasattr(response, 'text') and response.text:
            token_count = len(prompt.split())
            log_api_usage(action, token_count)
            log_api_call(user_id, action)
            cursor.execute("INSERT INTO api_cache (action, prompt, response) VALUES (%s, %s, %s)", (action, prompt, response.text))
            conn.commit()
            cursor.close()
            DB_POOL.putconn(conn)
            logger.info(f"Generated and cached Gemini response for action: {action}")
            return response.text
        else:
            cursor.close()
            DB_POOL.putconn(conn)
            logger.warning("No valid response from Gemini API")
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        if 'cursor' in locals():
            cursor.close()
        DB_POOL.putconn(conn)
        log_api_usage(f"{action}_Error", 0)
        logger.error(f"Gemini API error: {e}")
        return f"API Error: {str(e)}"

# -------------------- Initialize Database --------------------
init_db()

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="ResumeSmartX - AI ATS", page_icon="📄", layout='wide')

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Login"

# Apply theme
def apply_theme():
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT theme FROM users WHERE username = %s", (st.session_state.username,))
        theme = cursor.fetchone()
        cursor.close()
        theme = theme[0] if theme else 'Light'
    except Exception:
        theme = 'Light'
    finally:
        DB_POOL.putconn(conn)
    css = """
    <style>
        body, .stApp {
            background-color: %s;
            color: %s;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            padding: 10px;
            background-color: %s;
            color: %s;
        }
        @media (max-width: 600px) {
            .stColumn {
                display: block;
                width: 100%;
            }
            .stTextInput, .stTextArea {
                width: 100% !important;
            }
        }
        .bottom-right {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background-color: %s;
            color: %s;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 14px;
        }
    </style>
    <div class="bottom-right"><b>Built by AI Team</b></div>
    """
    if theme == 'Dark':
        st.markdown(css % ('#1E1E1E', '#FFFFFF', '#4CAF50', '#FFFFFF', 'rgba(0, 0, 0, 0.7)', '#FFFFFF'), unsafe_allow_html=True)
    elif theme == 'Auto':
        st.markdown("""
        <style>
            @media (prefers-color-scheme: dark) {
                body, .stApp { background-color: #1E1E1E; color: #FFFFFF; }
                .stButton>button { background-color: #4CAF50; color: #FFFFFF; }
                .bottom-right { background-color: rgba(0, 0, 0, 0.7); color: #FFFFFF; }
            }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(css % ('#FFFFFF', '#000000', '#4CAF50', '#FFFFFF', 'rgba(0, 0, 0, 0.7)', '#FFFFFF'), unsafe_allow_html=True)
    else:
        st.markdown(css % ('#FFFFFF', '#000000', '#4CAF50', '#FFFFFF', 'rgba(0, 0, 0, 0.7)', '#FFFFFF'), unsafe_allow_html=True)

# Sidebar Navigation
try:
    st.sidebar.image("logo.png", width=200)
except FileNotFoundError:
    st.sidebar.warning("logo.png not found. Please add it to the project directory.")
st.sidebar.title("Navigation")

if not st.session_state.authenticated:
    st.session_state.selected_tab = st.sidebar.radio("Choose an Option", ["Login", "Register"])
else:
    st.session_state.selected_tab = st.sidebar.radio("Choose a Feature", [
        "🏆 Resume Analysis", "📚 Question Bank", "📊 Data Science", "🔲 Top 3 MNCs",
        "🛠 Debug Code", "🤖 Voice Agent", "📜 History", "📊 Dashboard",
        "📋 Job Tracker", "✍️ Resume Builder", "👤 Profile"
    ])
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark", "Auto"], key="theme")
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET theme = %s WHERE username = %s", (theme, st.session_state.username))
        conn.commit()
        cursor.close()
    except Exception as e:
        logger.error(f"Failed to update theme: {e}")
    finally:
        DB_POOL.putconn(conn)
    apply_theme()

# Login Page
if st.session_state.selected_tab == "Login" and not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Login to ResumeSmartX</h1>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        submit = st.form_submit_button("Login")
        if submit:
            if username:
                if login_user(username):
                    st.session_state.authenticated = True
                    st.session_state.selected_tab = "🏆 Resume Analysis"
                    st.rerun()
                else:
                    st.error("Login failed. Please check your username.")
            else:
                st.error("Please enter a username.")

# Registration Page
elif st.session_state.selected_tab == "Register":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Register for ResumeSmartX</h1>", unsafe_allow_html=True)
    try:
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            mobile_number = st.text_input("Mobile Number")
            submit = st.form_submit_button("Register")
            if submit:
                if len(username) < 3:
                    st.error("Username must be at least 3 characters.")
                elif not email or '@' not in email:
                    st.error("Please enter a valid email.")
                elif not mobile_number or not mobile_number.isdigit() or len(mobile_number) < 10:
                    st.error("Please enter a valid mobile number (at least 10 digits).")
                else:
                    if register_user(username, email, mobile_number):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.selected_tab = "🏆 Resume Analysis"
                        st.success("Registration successful! Logged in.")
                        st.rerun()
                    else:
                        st.error("Registration failed. Username may already exist.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"Error: {str(e)}")

# Main App
elif st.session_state.authenticated:
    st.sidebar.write(f"Welcome, {st.session_state.username}!")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.selected_tab = "Login"
        st.rerun()

    if st.session_state.selected_tab == "🏆 Resume Analysis":
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>MY PERSONAL ATS</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)
        try:
            col1, col2 = st.columns([1, 1])
            with col1:
                input_text = st.text_area("📋 Job Description:", key="input", height=150)
            with col2:
                version_label = st.text_input("Resume Version Label", "Default Version")
                uploaded_file = st.file_uploader("📄 Upload your resume (PDF)...", type=['pdf'])
                if uploaded_file:
                    st.success("✅ Successfully uploaded.")
                    resume_text = ""
                    try:
                        reader = PdfReader(uploaded_file)
                        for page in reader.pages:
                            if page and page.extract_text():
                                resume_text += page.extract_text()
                        st.session_state['resume_text'] = resume_text
                        save_resume_to_postgres(uploaded_file.name, resume_text, version_label)
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")
                        logger.error(f"Failed to read PDF: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("<h3 style='text-align: center; margin-bottom: 2rem;'>🛠 Quick Actions</h3>", unsafe_allow_html=True)

        if st.button("📖 Tell Me About the Resume"):
            with st.spinner("Analyzing..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Please review the following resume and provide a detailed evaluation:\n\n{st.session_state['resume_text']}",
                        action="Tell_me_about_resume"
                    )
                    log_to_postgres("Tell_me_about_resume", response)
                    st.write(response)
                    st.session_state['resume_response'] = response
                    st.download_button("💾 Download Resume Evaluation", response, "resume_evaluation.txt")
                    if st.button("🔊 Read Resume Summary"):
                        with st.spinner("Generating audio..."):
                            try:
                                short_text = response[:2000]
                                audio_stream = client.generate(
                                    text=short_text,
                                    voice="Rachel",
                                    model="eleven_multilingual_v2",
                                    stream=True
                                )
                                audio_chunks = []
                                for chunk in audio_stream:
                                    if chunk:
                                        audio_chunks.append(chunk)
                                with open("resume_summary.mp3", "wb") as file:
                                    for chunk in audio_chunks:
                                        file.write(chunk)
                                st.success("Audio summary created successfully!")
                                st.audio("resume_summary.mp3")
                                logger.info("Generated audio summary")
                            except Exception as e:
                                st.error(f"Failed to generate audio: {e}")
                                logger.error(f"Failed to generate audio: {e}")

        if st.button("📊 Percentage Match"):
            with st.spinner("Analyzing..."):
                if not st.session_state.get('resume_text') or not input_text:
                    st.warning("Please upload a resume and provide a job description.")
                else:
                    response = get_gemini_response(
                        f"Evaluate the following resume against this job description and provide a percentage match first:\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state['resume_text']}",
                        action="Percentage_Match"
                    )
                    log_to_postgres("Percentage_Match", response)
                    st.write(response)
                    st.download_button("💾 Download Percentage Match", response, "percentage_match.txt")

        learning_path_duration = st.selectbox("📆 Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])
        if st.form_submit_button("🎓 Generate Learning Path"):
            with st.form("learning_path_form"):
                if not st.session_state.get('resume_text') or not input_text:
                    st.warning("Please upload a resume and provide a job description.")
                else:
                    response = get_gemini_response(
                        f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\n{input_text}\n\nResume:\n{st.session_state['resume_text']}",
                        action="Learning_Path"
                    )
                    log_to_postgres("Learning_Path", response)
                    st.write(response)
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    styles.add(ParagraphStyle(name='Custom', spaceAfter=12))
                    story = [Paragraph(f"Personalized Learning Path ({learning_path_duration} Months)", styles['Title'])]
                    for line in response.split('\n'):
                        story.append(Paragraph(line, styles['Custom']))
                        story.append(Spacer(1, 12))
                    doc.build(story)
                    st.download_button(
                        "💾 Download Learning Path PDF",
                        pdf_buffer.getvalue(),
                        f"learning_path_{learning_path_duration.lower().replace(' ', '_')}.pdf",
                        "application/pdf"
                    )

        if st.button("📝 Generate Updated Resume"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Suggest improvements and generate an updated resume for this candidate according to the job description:\n{st.session_state['resume_text']}",
                        action="Generate_Updated_Resume"
                    )
                    log_to_postgres("Generate_Updated_Resume", response)
                    st.write(response)
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = [Paragraph(response.replace('\n', '<br />'), styles['Normal'])]
                    doc.build(story)
                    st.download_button(
                        label="📝 Download Updated Resume",
                        data=pdf_buffer.getvalue(),
                        file_name="updated_resume.pdf",
                        mime="application/pdf"
                    )

        if st.button("❓ Generate 30 Interview Questions and Answers"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Generate 30 technical interview questions and their detailed answers based on the resume:\n{st.session_state['resume_text']}",
                        action="Interview_Questions"
                    )
                    log_to_postgres("Interview_Questions", response)
                    st.write(response)

        if st.button("🚖 Skill Development Plan"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text') or not input_text:
                    st.warning("Please upload a resume and provide a job description.")
                else:
                    response = get_gemini_response(
                        f"Based on the resume and job description, suggest courses, books, and projects to improve the person's weak or missing skills.\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state['resume_text']}",
                        action="Skill_Development"
                    )
                    log_to_postgres("Skill_Development", response)
                    st.write(response)

        if st.button("🎥 Mock Interview Questions"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text') or not input_text:
                    st.warning("Please upload a resume and provide a job description.")
                else:
                    response = get_gemini_response(
                        f"Generate follow-up interview questions based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state['resume_text']}",
                        action="Mock_Interview"
                    )
                    log_to_postgres("Mock_Interview", response)
                    st.write(response)

        if st.button("💡 AI Insights"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Based on this resume, suggest specific job roles that the candidate is best suited for and analyze market trends for their skills:\n\nResume:\n{st.session_state['resume_text']}",
                        action="AI_Insights"
                    )
                    log_to_postgres("AI_Insights", response)
                    try:
                        insights = json.loads(response)
                        st.write("📋 Recommendations:")
                        st.write(insights.get("job_roles", "No recommendations found."))
                        st.write("📈 Market Trends:")
                        st.write(insights.get("market_trends", "No trends available."))
                    except json.JSONDecodeError:
                        st.write("📋 AI Insights:")
                        st.write(response)

    elif st.session_state.selected_tab == "🔲 Top 3 MNCs":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>🚀 Top 3 MNCs for Data Science</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border: none; border-bottom: 2px solid #4CAF50; margin-bottom: 2rem;'>", unsafe_allow_html=True)
        try:
            if "selected_mnc" not in st.session_state:
                st.session_state["selected_mnc"] = None
            mnc_data = [
                {"name": "TCS", "color": "#FFA500", "icon": "🎯"},
                {"name": "Infosys", "color": "#FF0000", "icon": "🚀"},
                {"name": "Wipro", "color": "#800080", "icon": "🔍"}
            ]
            col1, col2, col3 = st.columns(3)
            for col, mnc in zip([col1, col2, col3], mnc_data):
                with col:
                    if st.button(f"{mnc['icon']} {mnc['name']}", key=f"{mnc['name']}_mnc"):
                        st.session_state["selected_mnc"] = mnc["name"]
            if st.session_state.get("selected_mnc"):
                selected_mnc = st.session_state["selected_mnc"]
                st.markdown(f"<h3 style='color: #FFA500; text-align: center;'>{selected_mnc} Data Science Prep</h3>", unsafe_allow_html=True)
                st.markdown("---")
                with st.spinner("Analyzing..."):
                    if not st.session_state.get('resume_text'):
                        st.warning("Please upload a resume first.")
                    else:
                        response = get_gemini_response(
                            f"Based on the candidate's resume, what additional skills and knowledge are needed to secure a Data Science role at {selected_mnc}?",
                            action="MNC_Skills"
                        )
                        log_to_postgres("MNC_Skills", response)
                        st.write(response)
                if st.button("📂 Project Types & Skills"):
                    with st.spinner("Loading..."):
                        if not st.session_state.get('resume_text'):
                            st.warning("Please upload a resume first.")
                        else:
                            response = get_gemini_response(
                                f"What types of data science projects does {selected_mnc} typically work on, and what skills are required?",
                                action="MNC_Projects"
                            )
                            log_to_postgres("MNC_Projects", response)
                            st.write(response)
                if st.button("🛠 Required Skills"):
                    with st.spinner("Loading..."):
                        if not st.session_state.get('resume_text'):
                            st.warning("Please upload a resume first.")
                        else:
                            response = get_gemini_response(
                                f"What technical and soft skills are required for a Data Science role at {selected_mnc}?",
                                action="MNC_Required_Skills"
                            )
                            log_to_postgres("MNC_Required_Skills", response)
                            st.write(response)
                if st.button("💡 Career Recommendations"):
                    with st.spinner("Loading..."):
                        if not st.session_state.get('resume_text'):
                            st.warning("Please upload a resume first.")
                        else:
                            response = get_gemini_response(
                                f"Based on the candidate's resume, what areas should they focus on to improve their chances for a Data Science role at {selected_mnc}?",
                                action="MNC_Career_Recs"
                            )
                            log_to_postgres("MNC_Career_Recs", response)
                            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Error in MNC section: {e}")

    elif st.session_state.selected_tab == "📊 Data Science":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📊 DSA & Data Science</h2>", unsafe_allow_html=True)
        try:
            level = st.selectbox("Select Difficulty Level:", ["Beginner", "Intermediate", "Advanced"])
            if st.button(f"Generate {level} DSA Questions"):
                with st.spinner("Generating..."):
                    response = get_gemini_response(
                        f"Generate 10 {level} DSA questions and answers for Data Science.",
                        action="DSA_Questions"
                    )
                    log_to_postgres("DSA_Questions", response)
                    st.write(response)
            topic = st.selectbox("Select DSA Topic:", [
                "Arrays", "Linked Lists", "Trees", "Graphs",
                "Dynamic Programming", "Sorting", "Searching", "Recursion"
            ])
            if st.button(f"Learn {topic} with Case Studies"):
                with st.spinner("Generating..."):
                    response = get_gemini_response(
                        f"Explain {topic} in simple terms for Data Science, including Python code examples and a real-world case study.",
                        action="DSA_Learn"
                    )
                    log_to_postgres("DSA_Learn", response)
                    st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Error in Data Science section: {e}")

    elif st.session_state.selected_tab == "📚 Question Bank":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📚 Question Bank</h2>", unsafe_allow_html=True)
        try:
            question_category = st.selectbox("Select Category:", [
                "Python", "Machine Learning", "Deep Learning", "SQL",
                "Data Warehousing", "Data Pipelines", "Docker"
            ])
            if st.button(f"Generate 30 {question_category} Questions"):
                with st.spinner("Generating..."):
                    response = get_gemini_response(
                        f"Generate 30 {question_category} interview questions with detailed answers.",
                        action="Question_Bank"
                    )
                    log_to_postgres("Question_Bank", response)
                    st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Error in Question Bank section: {e}")

    elif st.session_state.selected_tab == "🛠 Debug Code":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>🛠 Debug Code</h2>", unsafe_allow_html=True)
        try:
            code = st.text_area("Paste your Python code:", height=300)
            if st.button("Debug Code"):
                if not code.strip():
                    st.warning("Please enter some code.")
                else:
                    with st.spinner("Debugging..."):
                        response = get_gemini_response(
                            f"Debug the following Python code and provide a fixed version with explanations:\n\n```python\n{code}\n```",
                            action="Debug_Code"
                        )
                        log_to_postgres("Debug_Code", response)
                        st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Error in Debug Code section: {e}")

    elif st.session_state.selected_tab == "🤖 Voice Agent":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>🤖 Voice Agent</h2>", unsafe_allow_html=True)
        try:
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <a href='https://elevenlabs.io/app/talk-to?agent_id={AGENT_ID}' target='_blank'>
                        <button style='padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 8px;'>
                            Launch Voice Interview
                        </button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Error in Voice Agent section: {e}")

    elif st.session_state.selected_tab == "📜 History":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📜 History</h2>", unsafe_allow_html=True)
        try:
            conn = DB_POOL.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
            user_id = cursor.fetchone()[0]
            cursor.execute("SELECT action, response, created_at FROM button_logs WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            logs = cursor.fetchall()
            cursor.execute("SELECT filename, version_label, version_number, uploaded_at FROM resumes WHERE user_id = %s ORDER BY uploaded_at DESC", (user_id,))
            resumes = cursor.fetchall()
            cursor.close()
            DB_POOL.putconn(conn)
            st.subheader("Activity Logs")
            for log in logs:
                st.write(f"**{log[2].strftime('%Y-%m-%d %H:%M:%S')}**: {log[0]}")
                st.text_area("Response", log[1], height=200, disabled=True)
            st.subheader("Resumes")
            for resume in resumes:
                st.write(f"**{resume[3].strftime('%Y-%m-%d %H:%M:%S')}**: {resume[0]} ({resume[1]}, v{resume[2]})")
            if st.button("🔄 Revert to Previous Resume"):
                resume_options = [f"{r[0]} (v{r[2]})" for r in resumes]
                selected_resume = st.selectbox("Select Resume", resume_options)
                if selected_resume:
                    selected_filename = selected_resume.split(" (")[0]
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    cursor.execute("SELECT resume_text FROM resumes WHERE filename = %s AND user_id = %s ORDER BY version_number DESC LIMIT 1", (selected_filename, user_id))
                    resume_text = cursor.fetchone()
                    if resume_text:
                        st.session_state['resume_text'] = resume_text[0]
                        st.success("Successfully reverted to selected resume!")
                    cursor.close()
                    DB_POOL.putconn(conn)
        except Exception as e:
            st.error(f"Failed to load history: {e}")
            logger.error(f"Failed to load history: {e}")

    elif st.session_state.selected_tab == "📊 Dashboard":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📊 Career Dashboard</h2>", unsafe_allow_html=True)
        try:
            conn = DB_POOL.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
            user_id = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM resumes WHERE user_id = %s", (user_id,))
            resume_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM button_logs WHERE user_id = %s AND action LIKE '%Gemini%'", (user_id,))
            api_calls = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM api_usage WHERE user_id = %s AND timestamp > %s", (user_id, datetime.now() - timedelta(days=1)))
            daily_usage = cursor.fetchone()[0]
            cursor.execute("SELECT goal, status FROM learning_goals WHERE user_id = %s", (user_id,))
            goals = cursor.fetchall()
            cursor.execute("SELECT job_title, company, description, apply_link, created_at FROM job_alerts WHERE user_id = %s ORDER BY created_at DESC LIMIT 5", (user_id,))
            job_alerts = cursor.fetchall()
            cursor.close()
            DB_POOL.putconn(conn)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Resumes", resume_count)
                st.metric("Daily API Usage", f"{daily_usage}/50")
            with col2:
                st.metric("API Calls", api_calls)
                st.metric("Learning Goals", len(goals))
            if st.button("Export Dashboard Data"):
                data = {
                    "Resumes": [resume_count],
                    "API Calls": [api_calls],
                    "Daily Usage": [daily_usage],
                    "Goals": [len(goals)]
                }
                df = pd.DataFrame(data)
                csv_data = df.to_csv(index=False)
                st.download_button("Download CSV", csv_data, "dashboard_data.csv", "text/csv")
            st.subheader("Skill Gap Analysis")
            if resume_count > 0 and st.session_state.get('resume_text'):
                response = get_gemini_response(
                    f"Analyze the resume for skill gaps against current data science trends:\n{st.session_state['resume_text']}",
                    action="Skill_Gap_Analysis"
                )
                st.write(response)
            st.subheader("Learning Goals")
            for goal in goals:
                st.write(f"- {goal[0]} (Status: {goal[1]})")
            with st.form("add_goal"):
                new_goal = st.text_input("New Goal")
                submit = st.form_submit_button("Add Goal")
                if submit and new_goal:
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO learning_goals (user_id, goal) VALUES (%s, %s)", (user_id, new_goal))
                    conn.commit()
                    cursor.close()
                    DB_POOL.putconn(conn)
                    st.success("Goal added!")
                    st.rerun()
            st.subheader("Job Alerts")
            for alert in job_alerts:
                st.write(f"**{alert[4].strftime('%Y-%m-%d')}**: {alert[0]} at {alert[1]}")
                st.write(alert[2])
                st.markdown(f"[Apply]({alert[3]})")
            if st.button("🔍 Refresh Job Alerts"):
                with st.spinner("Refreshing..."):
                    try:
                        response = get_gemini_response(
                            f"Based on the resume, suggest 5 data science job opportunities with titles, companies, descriptions, and apply links:\n{st.session_state['resume_text']}",
                            action="Job_Alerts"
                        )
                        try:
                            jobs = json.loads(response)
                            conn = DB_POOL.getconn()
                            cursor = conn.cursor()
                            for job in jobs:
                                cursor.execute(
                                    "INSERT INTO job_alerts (user_id, job_title, company, description, apply_link) VALUES (%s, %s, %s, %s, %s)",
                                    (user_id, job.get("title", ""), job.get("company", ""), job.get("description", ""), job.get("apply_link", ""))
                                )
                            conn.commit()
                            cursor.close()
                            DB_POOL.putconn(conn)
                            st.success("Successfully updated job alerts!")
                            st.rerun()
                        except json.JSONDecodeError:
                            st.error(f"Failed to parse job alerts: {response}")
                            logger.error(f"Failed to parse job alerts: JSONDecodeError")
                    except Exception as e:
                        st.error(f"Failed to update job alerts: {e}")
                        logger.error(f"Failed to update job alerts: {e}")
            st.subheader("Industry News")
            st.write("- [Towards Data Science](https://towardsdatascience.com)")
            st.write("- [KDnuggets](https://www.kdnuggets.com)")
        except Exception as e:
            st.error(f"Failed to load dashboard: {e}")
            logger.error(f"Failed to load dashboard: {e}")

    elif st.session_state.selected_tab == "📋 Job Tracker":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📋 Job Tracker</h2>", unsafe_allow_html=True)
        try:
            with st.form("job_tracker"):
                company_name = st.text_input("Company")
                job_role = st.text_input("Role")
                application_date = st.date_input("Application Date")
                status = st.selectbox("Status", ["Applied", "Interviewing", "Offer", "Rejected"])
                resume_options = [(None, "None")]
                conn = DB_POOL.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
                user_id = cursor.fetchone()[0]
                cursor.execute("SELECT id, version_label FROM resumes WHERE user_id = %s", (user_id,))
                resumes = cursor.fetchall()
                cursor.close()
                DB_POOL.putconn(conn)
                resume_options.extend([(r[0], r[1]) for r in resumes])
                resume_id = st.selectbox("Select Resume", options=[r[0] for r in resume_options], format_func=lambda x: next((r[1] for r in resume_options if r[0] == x), "None"))
                submit = st.form_submit_button("Submit")
                if submit:
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO job_applications (user_id, company_name, job_role, application_date, status, resume_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        user_id,
                        company_name,
                        job_role,
                        application_date,
                        status,
                        resume_id if resume_id else None
                    ))
                    conn.commit()
                    cursor.close()
                    DB_POOL.putconn(conn)
                    st.success("Application added")
                    logger.info(f"Added job application for {st.session_state.username}")
            st.subheader("Applications")
            conn = DB_POOL.getconn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT company_name, job_role, application_date, status, resume_id
                FROM job_applications
                WHERE user_id = %s
                ORDER BY created_at DESC
            """, (user_id,))
            apps = cursor.fetchall()
            cursor.close()
            DB_POOL.putconn(conn)
            for app in apps:
                resume_label = "None"
                if app[4]:
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    cursor.execute("SELECT version_label FROM resumes WHERE id = %s", (app[4],))
                    result = cursor.fetchone()
                    resume_label = result[0] if result else "None"
                    cursor.close()
                    DB_POOL.putconn(conn)
                st.markdown(f"**{app[2]}**: {app[0]} ({app[1]}) - Status: {app[3]} - Resume: {resume_label}")
        except Exception as e:
            st.error(f"Failed to load job tracker: {e}")
            logger.error(f"Failed to load job tracker: {e}")

    elif st.session_state.selected_tab == "✍️ Resume Builder":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>✍️ Resume Builder</h2>", unsafe_allow_html=True)
        try:
            template = st.selectbox("Template", ["Chronological", "Functional"])
            with st.form("resume_form"):
                personal_info = st.text_area("Personal Info", height=100)
                education = st.text_area("Education", height=100)
                experience = st.text_area("Experience", height=100)
                skills = st.text_area("Skills", height=100)
                version_label = st.text_input("Version Label", "New Resume")
                submit = st.form_submit_button("Generate Resume")
                if submit:
                    resume_text = f"""
Personal Info:
{personal_info}

Education:
{education}

Experience:
{experience}

Skills:
{skills}
                    """
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    if template == "Chronological":
                        story.extend([
                            Paragraph(personal_info.replace('\n', '<br />'), styles['Title']),
                            Spacer(1, 12),
                            Paragraph("Education", styles['Heading2']),
                            Paragraph(education.replace('\n', '<br />'), styles['Normal']),
                            Spacer(1, 12),
                            Paragraph("Experience", styles['Heading2']),
                            Paragraph(experience.replace('\n', '<br />'), styles['Normal']),
                            Spacer(1, 12),
                            Paragraph("Skills", styles['Heading2']),
                            Paragraph(skills.replace('\n', '<br />'), styles['Normal'])
                        ])
                    else:
                        story.extend([
                            Paragraph(personal_info.replace('\n', '<br />'), styles['Title']),
                            Spacer(1, 12),
                            Paragraph("Skills", styles['Heading2']),
                            Paragraph(skills.replace('\n', '<br />'), styles['Normal']),
                            Spacer(1, 12),
                            Paragraph("Experience", styles['Heading2']),
                            Paragraph(experience.replace('\n', '<br />'), styles['Normal']),
                            Spacer(1, 12),
                            Paragraph("Education", styles['Heading2']),
                            Paragraph(education.replace('\n', '<br />'), styles['Normal'])
                        ])
                    doc.build(story)
                    st.session_state['resume_text'] = resume_text
                    resume_id = save_resume_to_postgres(f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", resume_text, version_label)
                    if resume_id:
                        st.download_button(
                            "Download Resume",
                            pdf_buffer.getvalue(),
                            "resume.pdf",
                            "application/pdf"
                        )
                        st.success("Resume generated!")
                    else:
                        st.error("Failed to save resume")
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Error in Resume Builder: {e}")

    elif st.session_state.selected_tab == "👤 Profile":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>👤 Profile</h2>", unsafe_allow_html=True)
        try:
            conn = DB_POOL.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT email, mobile_number, theme FROM users WHERE username = %s", (st.session_state.username,))
            user_data = cursor.fetchone()
            cursor.close()
            DB_POOL.putconn(conn)
            with st.form("profile_form"):
                email = st.text_input("Email", value=user_data[0] if user_data else "")
                mobile_number = st.text_input("Mobile Number", value=user_data[1] if user_data else "")
                submit = st.form_submit_button("Update Profile")
                if submit:
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    try:
                        cursor.execute("UPDATE users SET email = %s, mobile_number = %s WHERE username = %s", (email, mobile_number, st.session_state.username))
                        conn.commit()
                        st.success("Profile updated successfully!")
                        logger.info(f"Updated profile for {st.session_state.username}")
                    except psycopg2.IntegrityError:
                        st.error("Email already in use.")
                        logger.error(f"Failed to update profile: Email already in use")
                    except Exception as e:
                        st.error(f"Failed to update profile: {e}")
                        logger.error(f"Failed to update profile: {e}")
                    finally:
                        cursor.close()
                        DB_POOL.putconn(conn)
        except Exception as e:
            st.error(f"Failed to load profile: {e}")
            logger.error(f"Failed to load profile: {e}")

else:
    st.error("Please log in to access the app.")
    logger.warning("User not authenticated")

logger.info("Application completed successfully")