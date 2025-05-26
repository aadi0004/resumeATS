import logging
import io
import os
import json
import threading
import csv
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
from twilio.rest import Client

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
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
VERIFY_SERVICE_SID = os.getenv("VERIFY_SERVICE_SID")

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

# Initialize Twilio client
try:
    twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {e}")
    st.error("Twilio initialization failed. Please check your TWILIO_SID and TWILIO_AUTH_TOKEN.")

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
                password TEXT,
                phone_number VARCHAR(15),
                theme VARCHAR(10) DEFAULT 'Light',
                otp_code VARCHAR(6),
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

def register_user(username: str, phone_number: str) -> bool:
    conn = DB_POOL.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, phone_number) VALUES (%s, %s)", (username, phone_number))
        conn.commit()
        cursor.close()
        logger.info(f"User registered: {username}")
        return True
    except psycopg2.IntegrityError:
        st.error("Username already exists. Please choose a different username.")
        logger.warning(f"Registration failed for {username}: Username exists")
        return False
    except Exception as e:
        st.error(f"Registration failed: {e}")
        logger.error(f"Registration failed for {username}: {e}")
        return False
    finally:
        DB_POOL.putconn(conn)

def send_otp(phone_number: str) -> bool:
    try:
        verification = twilio_client.verify.v2.services(VERIFY_SERVICE_SID).verifications.create(
            to=phone_number, channel="sms"
        )
        if verification.status == "pending":
            logger.info(f"OTP sent to {phone_number}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to send OTP to {phone_number}: {e}")
        st.error(f"Failed to send OTP: {e}")
        return False

def verify_otp(phone_number: str, otp_code: str) -> bool:
    try:
        verification_check = twilio_client.verify.v2.services(VERIFY_SERVICE_SID).verification_checks.create(
            to=phone_number, code=otp_code
        )
        if verification_check.status == "approved":
            conn = DB_POOL.getconn()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET otp_code = NULL WHERE phone_number = %s", (phone_number,))
            conn.commit()
            cursor.close()
            DB_POOL.putconn(conn)
            logger.info(f"OTP verified for {phone_number}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to verify OTP for {phone_number}: {e}")
        st.error(f"Invalid OTP: {e}")
        return False

def login_user(phone_number: str, otp_code: str) -> bool:
    if verify_otp(phone_number, otp_code):
        conn = DB_POOL.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE phone_number = %s", (phone_number,))
            result = cursor.fetchone()
            cursor.close()
            if result:
                logger.info(f"User logged in: {result[0]}")
                return True
            else:
                st.error("Phone number not registered.")
                logger.warning(f"Login failed for {phone_number}: Not registered")
                return False
        except Exception as e:
            st.error(f"Login failed: {e}")
            logger.error(f"Login failed for {phone_number}: {e}")
            return False
        finally:
            DB_POOL.putconn(conn)
    return False

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
        st.error(f"Failed to save resume: {e}")
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
        logger.warning("Empty prompt provided to Gemini API")
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
if 'otp_sent' not in st.session_state:
    st.session_state.otp_sent = False
if 'phone_number' not in st.session_state:
    st.session_state.phone_number = None

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
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 14px;
        }
    </style>
    <div class="bottom-right"><b>Built by AI Team</b></div>
    """
    if theme == 'Dark':
        st.markdown(css % ('#1E1E1E', '#FFFFFF'), unsafe_allow_html=True)
    elif theme == 'Auto':
        st.markdown("""
        <style>
            @media (prefers-color-scheme: dark) {
                body, .stApp { background-color: #1E1E1E; color: #FFFFFF; }
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown(css % ('#FFFFFF', '#000000'), unsafe_allow_html=True)

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
        "🏆 Resume Analysis", "📚 Question Bank", "📊 DSA & Data Science", "🔝 Top 3 MNCs",
        "🛠️ Code Debugger", "🤖 Voice Agent Chat", "📜 History", "📊 Dashboard",
        "📋 Job Tracker", "✍️ Resume Builder"
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
    if not st.session_state.otp_sent:
        with st.form("login_form"):
            phone_number = st.text_input("Phone Number (e.g., +919876543210)")
            submit = st.form_submit_button("Send OTP")
            if submit:
                if phone_number and phone_number.startswith("+") and len(phone_number) >= 10:
                    if send_otp(phone_number):
                        st.session_state.otp_sent = True
                        st.session_state.phone_number = phone_number
                        st.success("OTP sent to your phone number!")
                        st.rerun()
                    else:
                        st.error("Failed to send OTP. Please try again.")
                else:
                    st.error("Please enter a valid phone number.")
    else:
        with st.form("otp_form"):
            otp_code = st.text_input("Enter OTP", type="password")
            submit = st.form_submit_button("Verify OTP")
            if submit:
                if login_user(st.session_state.phone_number, otp_code):
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    cursor.execute("SELECT username FROM users WHERE phone_number = %s", (st.session_state.phone_number,))
                    st.session_state.username = cursor.fetchone()[0]
                    cursor.close()
                    DB_POOL.putconn(conn)
                    st.session_state.authenticated = True
                    st.session_state.selected_tab = "🏆 Resume Analysis"
                    st.session_state.otp_sent = False
                    st.rerun()
                else:
                    st.error("Invalid OTP or phone number not registered.")

# Registration Page
elif st.session_state.selected_tab == "Register" and not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Register for ResumeSmartX</h1>", unsafe_allow_html=True)
    if not st.session_state.otp_sent:
        with st.form("register_form"):
            username = st.text_input("Username")
            phone_number = st.text_input("Phone Number (e.g., +919876543210)")
            submit = st.form_submit_button("Send OTP")
            if submit:
                if len(username) < 3:
                    st.error("Username must be at least 3 characters.")
                elif not phone_number or not phone_number.startswith("+") or len(phone_number) < 10:
                    st.error("Please enter a valid phone number.")
                else:
                    if send_otp(phone_number):
                        st.session_state.otp_sent = True
                        st.session_state.phone_number = phone_number
                        st.session_state.temp_username = username
                        st.success("OTP sent to your phone number!")
                        st.rerun()
                    else:
                        st.error("Failed to send OTP. Please try again.")
    else:
        with st.form("otp_register_form"):
            otp_code = st.text_input("Enter OTP", type="password")
            submit = st.form_submit_button("Verify OTP")
            if submit:
                if verify_otp(st.session_state.phone_number, otp_code):
                    if register_user(st.session_state.temp_username, st.session_state.phone_number):
                        st.session_state.authenticated = True
                        st.session_state.username = st.session_state.temp_username
                        st.session_state.selected_tab = "🏆 Resume Analysis"
                        st.session_state.otp_sent = False
                        st.success("Registration successful! Logged in.")
                        st.rerun()
                else:
                    st.error("Invalid OTP.")

# Main App
elif st.session_state.authenticated:
    st.sidebar.write(f"Welcome, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.selected_tab = "Login"
        st.session_state.otp_sent = False
        st.rerun()

    if st.session_state.selected_tab == "🏆 Resume Analysis":
        st.markdown("""
            <h1 style='text-align: center; color: #4CAF50;'>MY PERSONAL ATS</h1>
            <hr style='border: 1px solid #4CAF50;'>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            input_text = st.text_area("📋 Job Description:", key="input", height=150)
        with col2:
            version_label = st.text_input("Resume Version Label", "Default Version")
            uploaded_file = st.file_uploader("📄 Upload your resume (PDF)...", type=['pdf'])
            if uploaded_file:
                st.success("✅ PDF Uploaded Successfully.")
                resume_text = ""
                try:
                    reader = PdfReader(uploaded_file)
                    for page in reader.pages:
                        if page and page.extract_text():
                            resume_text += page.extract_text()
                    st.session_state['resume_text'] = resume_text
                    save_resume_to_postgres(uploaded_file.name, resume_text, version_label)
                except Exception as e:
                    st.error(f"Failed to read PDF: {e}")
                    logger.error(f"Failed to read PDF: {e}")
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🛠 Quick Actions</h3>", unsafe_allow_html=True)

        if st.button("📖 Tell Me About the Resume"):
            with st.spinner("Analyzing..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Please review the following resume and provide a detailed evaluation: {st.session_state['resume_text']}",
                        action="Tell_me_about_resume"
                    )
                    log_to_postgres("Tell_me_about_resume", response)
                    st.session_state['resume_response'] = response
                    st.write(response)
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
                                with open("resume_summary.mp3", "wb") as f:
                                    for chunk in audio_chunks:
                                        f.write(chunk)
                                st.success("Audio summary created!")
                                st.audio("resume_summary.mp3")
                                logger.info("Generated audio summary")
                            except Exception as e:
                                st.error(f"Audio generation failed: {e}")
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
        if st.button("🎓 Personalized Learning Path"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text') or not input_text:
                    st.warning("Please upload a resume and provide a job description.")
                else:
                    response = get_gemini_response(
                        f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state['resume_text']} and also suggest books and other important things",
                        action="Personalized_Learning_Path"
                    )
                    log_to_postgres("Personalized_Learning_Path", response)
                    st.write(response)
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    styles.add(ParagraphStyle(name='Custom', spaceAfter=12))
                    story = [Paragraph(f"Personalized Learning Path ({learning_path_duration})", styles['Title']), Spacer(1, 12)]
                    for line in response.split('\n'):
                        story.append(Paragraph(line, styles['Custom']))
                        story.append(Spacer(1, 12))
                    doc.build(story)
                    st.download_button(
                        f"💾 Download Learning Path PDF",
                        pdf_buffer.getvalue(),
                        f"learning_path_{learning_path_duration.replace(' ', '_').lower()}.pdf",
                        "application/pdf"
                    )

        if st.button("📝 Generate Updated Resume"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Suggest improvements and generate an updated resume for this candidate according to the job description, not more than 2 pages:\n{st.session_state['resume_text']}",
                        action="Generate_Updated_Resume"
                    )
                    log_to_postgres("Generate_Updated_Resume", response)
                    st.write(response)
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = [Paragraph(response.replace('\n', '<br/>'), styles['Normal'])]
                    doc.build(story)
                    st.download_button(
                        label="📥 Download Updated Resume",
                        data=pdf_buffer.getvalue(),
                        file_name="Updated_Resume.pdf",
                        mime="application/pdf"
                    )

        if st.button("❓ Generate 30 Interview Questions and Answers"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Generate 30 technical interview questions and their detailed answers based on the resume:\n{st.session_state['resume_text']}",
                        action="Generate_Interview_Questions"
                    )
                    log_to_postgres("Generate_Interview_Questions", response)
                    st.write(response)

        if st.button("🚖 Skill Development Plan"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text') or not input_text:
                    st.warning("Please upload a resume and provide a job description.")
                else:
                    response = get_gemini_response(
                        f"Based on the resume and job description, suggest courses, books, and projects to improve the person's weak or missing skills.\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state['resume_text']}",
                        action="Skill_Development_Plan"
                    )
                    log_to_postgres("Skill_Development_Plan", response)
                    st.write(response)

        if st.button("🎥 Mock Interview Questions"):
            with st.spinner("Generating..."):
                if not st.session_state.get('resume_text') or not input_text:
                    st.warning("Please upload a resume and provide a job description.")
                else:
                    response = get_gemini_response(
                        f"Generate follow-up interview questions based on the resume and job description, simulating a live interview.\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state['resume_text']}",
                        action="Mock_Interview_Questions"
                    )
                    log_to_postgres("Mock_Interview_Questions", response)
                    st.write(response)

        if st.button("💡 AI Insights"):
            with st.spinner("Analyzing..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Based on this resume, suggest specific job roles the user is most suited for and analyze market trends for their skills.\n\nResume:\n{st.session_state['resume_text']}",
                        action="AI_Driven_Insights"
                    )
                    log_to_postgres("AI_Driven_Insights", response)
                    try:
                        recommendations = json.loads(response)
                        st.write("📋 Smart Recommendations:")
                        st.write(recommendations.get("job_roles", "No recommendations found."))
                        st.write("📊 Market Trends:")
                        st.write(recommendations.get("market_trends", "No market trends available."))
                    except json.JSONDecodeError:
                        st.write("📋 AI-Driven Insights:")
                        st.write(response)

    elif st.session_state.selected_tab == "🔝 Top 3 MNCs":
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>🚀 MNC Data Science Preparation</h2>", unsafe_allow_html=True)
        st.markdown("---")
        if "selected_mnc" not in st.session_state:
            st.session_state["selected_mnc"] = None
        mnc_data = [
            {"name": "TCS", "color": "#FFA500", "icon": "🎯"},
            {"name": "Infosys", "color": "#03A9F4", "icon": "🚖"},
            {"name": "Wipro", "color": "#9C27B0", "icon": "🔍"},
        ]
        col1, col2, col3 = st.columns(3)
        for col, mnc in zip([col1, col2, col3], mnc_data):
            with col:
                if st.button(f"{mnc['icon']} {mnc['name']}", key=f"{mnc['name']}_button"):
                    st.session_state["selected_mnc"] = mnc["name"]
        if st.session_state["selected_mnc"]:
            selected_mnc = st.session_state["selected_mnc"]
            st.markdown(f"<h3 style='color: #FFA500; text-align: center;'>{selected_mnc} Data Science Preparation</h3>", unsafe_allow_html=True)
            st.markdown("---")
            with st.spinner("Analyzing..."):
                if not st.session_state.get('resume_text'):
                    st.warning("Please upload a resume first.")
                else:
                    response = get_gemini_response(
                        f"Based on the candidate's qualifications and resume, what additional skills and knowledge are needed to secure a Data Science role at {selected_mnc}?",
                        action="Additional_Skills_MNCS"
                    )
                    log_to_postgres("Additional_Skills_MNCS", response)
                    st.info(response)
            if st.button("📂 Project Types & Required Skills"):
                with st.spinner("Loading..."):
                    if not st.session_state.get('resume_text'):
                        st.warning("Please upload a resume first.")
                    else:
                        response = get_gemini_response(
                            f"What types of Data Science projects does {selected_mnc} typically work on, and what skills align best?",
                            action="Project_Types_Skills"
                        )
                        log_to_postgres("Project_Types_Skills", response)
                        st.success(response)
            if st.button("🛠 Required Skills"):
                with st.spinner("Loading..."):
                    if not st.session_state.get('resume_text'):
                        st.warning("Please upload a resume first.")
                    else:
                        response = get_gemini_response(
                            f"What key technical and soft skills are needed for a Data Science role at {selected_mnc}?",
                            action="Required_Skills"
                        )
                        log_to_postgres("Required_Skills", response)
                        st.success(response)
            if st.button("💡 Career Recommendations"):
                with st.spinner("Loading..."):
                    if not st.session_state.get('resume_text'):
                        st.warning("Please upload a resume first.")
                    else:
                        response = get_gemini_response(
                            f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at {selected_mnc}?",
                            action="Career_Recommendations"
                        )
                        log_to_postgres("Career_Recommendations", response)
                        st.success(response)

    elif st.session_state.selected_tab == "📊 DSA & Data Science":
        st.markdown("<h3 style='text-align: center;'>🛠 DSA for Data Science</h3>", unsafe_allow_html=True)
        level = st.selectbox("📚 Select Difficulty Level:", ["Beginner", "Intermediate", "Advanced"])
        if st.button(f"📝 Generate {level} DSA Questions"):
            with st.spinner("Generating..."):
                response = get_gemini_response(
                    f"Generate 10 DSA questions and answers for data science at {level} level.",
                    action="DSA_Questions"
                )
                log_to_postgres("DSA_Questions", response)
                st.write(response)
        topic = st.selectbox("🗂 Select DSA Topic:", [
            "Arrays", "Linked Lists", "Trees", "Graphs",
            "Dynamic Programming", "Recursion",
            "Algorithm Complexity (Big O Notation)", "Sorting", "Searching"
        ])
        if st.button(f"📖 Teach me {topic} with Case Studies"):
            with st.spinner("Gathering resources..."):
                explanation_response = get_gemini_response(
                    f"Explain the {topic} topic in an easy-to-understand way suitable for beginners, using simple language and clear examples. Add details like definition, examples of {topic}, and code implementation in Python with full explanation of the code.",
                    action="Teach_me_DSA_Topics"
                )
                log_to_postgres("Teach_me_DSA_Topics", explanation_response)
                st.write(explanation_response)
                case_study_response = get_gemini_response(
                    f"Provide a real-world case study on {topic} for data science/data engineering/ML/AI with a detailed, easy-to-understand solution.",
                    action="Case_Study_DSA_Topics"
                )
                log_to_postgres("Case_Study_DSA_Topics", case_study_response)
                st.write(case_study_response)

    elif st.session_state.selected_tab == "📚 Question Bank":
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📖 Question Bank</h2>", unsafe_allow_html=True)
        st.markdown("---")
        question_category = st.selectbox("❓ Select Question Category:", [
            "Python", "Machine Learning", "Deep Learning", "Docker",
            "Data Warehousing", "Data Pipelines", "Data Modeling", "SQL"
        ])
        if st.button(f"📖 Generate 30 {question_category} Interview Questions"):
            with st.spinner("Loading..."):
                response = get_gemini_response(
                    f"Generate 30 {question_category} interview questions and detailed answers.",
                    action="Interview_Questions"
                )
                log_to_postgres("Interview_Questions", response)
                st.write(response)

    elif st.session_state.selected_tab == "🛠️ Code Debugger":
        st.markdown("<h3 style='text-align: center;'>🛠 Python Code Debugger</h3>", unsafe_allow_html=True)
        user_code = st.text_area("Paste your Python code below:", height=300)
        if st.button("Check & Fix Code"):
            if not user_code.strip():
                st.warning("Please enter some code.")
            else:
                with st.spinner("Analyzing and fixing code..."):
                    prompt = (
                        f"Analyze the following Python code for bugs, syntax errors, and logic errors.\n"
                        f"If it has issues, correct them. Return the fixed code and briefly explain the changes made.\n\n"
                        f"Code:\n```python\n{user_code}\n```"
                    )
                    response = get_gemini_response(prompt, action="Code_Debugger")
                    log_to_postgres("Code_Debugger", response)
                    st.subheader("✅ Corrected Code")
                    st.code(response, language="python")

    elif st.session_state.selected_tab == "🤖 Voice Agent Chat":
        st.markdown("""
            <h1 style='text-align: center; color: #4CAF50;'>Talk to AI Interviewer 🤖🎤</h1>
            <hr style='border: 1px solid #4CAF50;'>
            <p style='text-align: center;'>Start a real-time voice conversation with our AI agent powered by ElevenLabs.</p>
            <div style='text-align: center;'>
                <a href='https://elevenlabs.io/app/talk-to?agent_id={AGENT_ID}' target='_blank'>
                    <button style='padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer;'>
                        🚖 Launch Voice Interview Agent
                    </button>
                </a>
            </div>
        """, unsafe_allow_html=True)

    elif st.session_state.selected_tab == "📜 History":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📜 Your Activity History</h2>", unsafe_allow_html=True)
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
            st.subheader("Past Actions")
            for log in logs:
                st.write(f"**{log[2].strftime('%Y-%m-%d %H:%M:%S')}** - {log[0]}")
                st.text_area("Response", log[1], height=100, disabled=True)
            st.subheader("Uploaded Resumes")
            for resume in resumes:
                st.write(f"**{resume[3].strftime('%Y-%m-%d %H:%M:%S')}** - {resume[0]} (Version: {resume[1]}, #{resume[2]})")
            if st.button("🔄 Revert to Previous Resume Version"):
                resume_id = st.selectbox("Select Resume Version", [f"{r[0]} (Version #{r[2]})" for r in resumes])
                if resume_id:
                    selected_filename = resume_id.split(" (")[0]
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    cursor.execute("SELECT resume_text FROM resumes WHERE filename = %s AND user_id = %s ORDER BY version_number DESC LIMIT 1", (selected_filename, user_id))
                    st.session_state['resume_text'] = cursor.fetchone()[0]
                    cursor.close()
                    DB_POOL.putconn(conn)
                    st.success("Reverted to selected resume version!")
        except Exception as e:
            st.error(f"Failed to fetch history: {e}")
            logger.error(f"Failed to fetch history: {e}")

    elif st.session_state.selected_tab == "📊 Dashboard":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📊 Your Career Dashboard</h2>", unsafe_allow_html=True)
        try:
            conn = DB_POOL.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
            user_id = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM resumes WHERE user_id = %s", (user_id,))
            resume_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM button_logs WHERE user_id = %s AND action LIKE '%%Gemini_API_Call%%'", (user_id,))
            api_calls = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(*) FROM api_usage WHERE user_id = %s AND timestamp > %s",
                (user_id, datetime.now() - timedelta(days=1))
            )
            daily_api_usage = cursor.fetchone()[0]
            cursor.execute("SELECT goal, status FROM learning_goals WHERE user_id = %s", (user_id,))
            goals = cursor.fetchall()
            cursor.execute("SELECT job_title, company, description, apply_link, created_at FROM job_alerts WHERE user_id = %s ORDER BY created_at DESC LIMIT 5", (user_id,))
            job_alerts = cursor.fetchall()
            cursor.close()
            DB_POOL.putconn(conn)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Resumes Uploaded", resume_count)
                st.metric("Daily API Usage", f"{daily_api_usage}/50")
            with col2:
                st.metric("API Calls Made", api_calls)
                st.metric("Learning Goals", len(goals))
            if st.button("📊 Export Dashboard Data"):
                data = {
                    "Resumes Uploaded": [resume_count],
                    "Total API Calls": [api_calls],
                    "Daily API Usage": [daily_api_usage],
                    "Learning Goals": [len(goals)]
                }
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "dashboard_data.csv", "text/csv")
            st.subheader("Skill Gap Analysis")
            if resume_count > 0 and st.session_state.get('resume_text'):
                response = get_gemini_response(
                    f"Analyze this resume against current data science job market trends and identify skill gaps:\n{st.session_state['resume_text']}",
                    action="Skill_Gap_Analysis"
                )
                st.info(response)
            st.subheader("Learning Goals")
            for goal in goals:
                st.write(f"- {goal[0]} (Status: {goal[1]})")
            with st.form("add_goal_form"):
                new_goal = st.text_input("Add New Learning Goal")
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
                st.write(f"**{alert[4].strftime('%Y-%m-%d')}** - {alert[0]} at {alert[1]}")
                st.write(alert[2])
                st.markdown(f"[Apply]({alert[3]})")
            if st.button("🔎 Refresh Job Alerts"):
                response = get_gemini_response(
                    f"Based on this resume, suggest 5 data science job opportunities with titles, companies, descriptions, and apply links:\n{st.session_state['resume_text']}",
                    action="Job_Alerts"
                )
                try:
                    jobs = json.loads(response)
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    for job in jobs:
                        cursor.execute(
                            "INSERT INTO job_alerts (user_id, job_title, company, description, apply_link) VALUES (%s, %s, %s, %s, %s)",
                            (user_id, job.get("title"), job.get("company"), job.get("description"), job.get("apply_link"))
                        )
                    conn.commit()
                    cursor.close()
                    DB_POOL.putconn(conn)
                    st.success("Job alerts updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update job alerts: {e}")
            st.subheader("Industry News")
            st.write("Latest Data Science News (Mock RSS Feed):")
            st.write("- [Towards Data Science: Top 5 ML Trends](https://towardsdatascience.com)")
            st.write("- [KDnuggets: AI in 2025](https://www.kdnuggets.com)")
        except Exception as e:
            st.error(f"Failed to load dashboard: {e}")
            logger.error(f"Failed to load dashboard: {e}")

    elif st.session_state.selected_tab == "📋 Job Tracker":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>📋 Job Application Tracker</h2>", unsafe_allow_html=True)
        with st.form("job_tracker_form"):
            company_name = st.text_input("Company Name")
            job_role = st.text_input("Job Role")
            application_date = st.date_input("Application Date")
            status = st.selectbox("Status", ["Applied", "Interviewing", "Offer", "Rejected"])
            resume_options = [(None, "None")]
            try:
                conn = DB_POOL.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
                user_id = cursor.fetchone()[0]
                cursor.execute("SELECT id, version_label FROM resumes WHERE user_id = %s", (user_id,))
                resumes = cursor.fetchall()
                resume_options.extend([(r[0], r[1]) for r in resumes])
                cursor.close()
                DB_POOL.putconn(conn)
            except Exception as e:
                st.error(f"Failed to fetch resumes: {e}")
                logger.error(f"Failed to fetch resumes: {e}")
            resume_id = st.selectbox("Select Resume", options=[r[0] for r in resume_options], format_func=lambda x: next((r[1] for r in resume_options if r[0] == x), "None"))
            submit = st.form_submit_button("Submit")
            if submit:
                try:
                    conn = DB_POOL.getconn()
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO job_applications (user_id, company_name, job_role, application_date, resume_id, status)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (user_id, company_name, job_role, application_date, resume_id if resume_id else None, status)
                    )
                    conn.commit()
                    cursor.close()
                    DB_POOL.putconn(conn)
                    st.success("Application added successfully!")
                    logger.info(f"Added job application for user: {st.session_state.username}")
                except Exception as e:
                    st.error(f"Failed to add application: {e}")
                    logger.error(f"Failed to add job application: {e}")
        st.subheader("Your Applications")
        try:
            conn = DB_POOL.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s", (st.session_state.username,))
            user_id = cursor.fetchone()[0]
            cursor.execute("""
                SELECT company_name, job_role, application_date, status, resume_id
                FROM job_applications
                WHERE user_id = %s
                ORDER BY application_date DESC
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
                    if result:
                        resume_label = result[0]
                    cursor.close()
                    DB_POOL.putconn(conn)
                st.write(f"**{app[2]}** - {app[0]} ({app[1]}) - Status: {app[3]} - Resume: {resume_label}")
            logger.info(f"Fetched job applications for user: {st.session_state.username}")
        except Exception as e:
            st.error(f"Failed to fetch applications: {e}")
            logger.error(f"Failed to fetch applications: {e}")

    elif st.session_state.selected_tab == "✍️ Resume Builder":
        st.markdown("<h2 style='text-align: center; color: #FFA500;'>✍️ Resume Builder</h2>", unsafe_allow_html=True)
        template = st.selectbox("Select Template", ["Chronological", "Functional"])
        with st.form("resume_builder_form"):
            personal_info = st.text_area("Personal Information (Name, Contact)", height=100)
            education = st.text_area("Education", height=100)
            experience = st.text_area("Work Experience", height=100)
            skills = st.text_area("Skills", height=100)
            version_label = st.text_input("Version Label", "Built Resume")
            submit = st.form_submit_button("Generate Resume")
            if submit:
                resume_text = f"""
                {personal_info}

                Education
                {education}

                Work Experience
                {experience}

                Skills
                {skills}
                """
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                if template == "Chronological":
                    story.extend([
                        Paragraph(personal_info.replace('\n', '<br/>'), styles['Title']),
                        Spacer(1, 12),
                        Paragraph("Education", styles['Heading2']),
                        Paragraph(education.replace('\n', '<br/>'), styles['Normal']),
                        Spacer(1, 12),
                        Paragraph("Work Experience", styles['Heading2']),
                        Paragraph(experience.replace('\n', '<br/>'), styles['Normal']),
                        Spacer(1, 12),
                        Paragraph("Skills", styles['Heading2']),
                        Paragraph(skills.replace('\n', '<br/>'), styles['Normal'])
                    ])
                else:  # Functional
                    story.extend([
                        Paragraph(personal_info.replace('\n', '<br/>'), styles['Title']),
                        Spacer(1, 12),
                        Paragraph("Skills", styles['Heading2']),
                        Paragraph(skills.replace('\n', '<br/>'), styles['Normal']),
                        Spacer(1, 12),
                        Paragraph("Work Experience", styles['Heading2']),
                        Paragraph(experience.replace('\n', '<br/>'), styles['Normal']),
                        Spacer(1, 12),
                        Paragraph("Education", styles['Heading2']),
                        Paragraph(education.replace('\n', '<br/>'), styles['Normal'])
                    ])
                doc.build(story)
                st.session_state['resume_text'] = resume_text
                resume_id = save_resume_to_postgres(f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", resume_text, version_label)
                if resume_id:
                    st.download_button(
                        label="📥 Download Resume",
                        data=pdf_buffer.getvalue(),
                        file_name="generated_resume.pdf",
                        mime="application/pdf"
                    )
                    st.success("Resume generated and saved!")
                else:
                    st.error("Failed to save resume.")

else:
    st.error("Please log in to access the application.")

logger.info("Application rendered successfully")