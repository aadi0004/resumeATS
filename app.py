import tempfile
import PyPDF2
import docx2txt
from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import json
import streamlit.components.v1 as components
import requests
import psycopg2
import bcrypt
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVEN_API_KEY)

# -------------------- ✅ LOGGING SETUP START --------------------
import csv
import threading

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

def log_api_usage(action, tokens_generated):
    with csv_lock:
        current_total = get_current_total_tokens()
        new_total = current_total + tokens_generated
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                action,
                1,
                tokens_generated,
                new_total
            ])

# -------------------- ✅ LOGGING SETUP END --------------------

# -------------------- ✅ DATABASE FUNCTIONS --------------------
def init_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            dbname=os.getenv("PG_DB")
        )
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS button_logs (
                id SERIAL PRIMARY KEY,
                action VARCHAR(255),
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255),
                resume_text TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")

def register_user(username, password):
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            dbname=os.getenv("PG_DB")
        )
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except psycopg2.IntegrityError:
        st.error("Username already exists. Please choose a different username.")
        return False
    except Exception as e:
        st.error(f"Registration failed: {e}")
        return False

def login_user(username, password):
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            dbname=os.getenv("PG_DB")
        )
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
            return True
        else:
            st.error("Invalid username or password.")
            return False
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False

def log_to_postgres(action, response):
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            dbname=os.getenv("PG_DB")
        )
        cursor = conn.cursor()
        cursor.execute("INSERT INTO button_logs (action, response) VALUES (%s, %s)", (action, response))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"PostgreSQL logging failed: {e}")

def save_resume_to_postgres(filename, resume_text):
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            dbname=os.getenv("PG_DB")
        )
        cursor = conn.cursor()
        cursor.execute("INSERT INTO resumes (filename, resume_text) VALUES (%s, %s)", (filename, resume_text))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"❌ Failed to save resume to PostgreSQL: {e}")

# -------------------- ✅ Gemini API Wrapper --------------------
def get_gemini_response(prompt, action="Gemini_API_Call"):
    if not prompt.strip():
        return "Error: Prompt is empty. Please provide a valid prompt."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            prompt,
            f"Add randomness: {os.urandom(8).hex()}"
        ])
        if hasattr(response, 'text') and response.text:
            token_count = len(prompt.split())
            log_api_usage(action, token_count)
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        log_api_usage(f"{action}_Error", 0)
        return f"API Error: {str(e)}"

# -------------------- ✅ Initialize Database --------------------
init_db()

# -------------------- ✅ Streamlit App --------------------
st.set_page_config(page_title="ResumeSmartX - AI ATS", page_icon="📄", layout='wide')

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Login"

# Sidebar Navigation
st.sidebar.image("logo.png", width=200)
st.sidebar.title("Navigation")

if not st.session_state.authenticated:
    st.session_state.selected_tab = st.sidebar.radio("Choose an Option", ["Login", "Register"])
else:
    st.session_state.selected_tab = st.sidebar.radio("Choose a Feature", [
        "🏆 Resume Analysis", "📚 Question Bank", "📊 DSA & Data Science", "🔝Top 3 MNCs", 
        "🛠️ Code Debugger", "🤖 Voice Agent Chat"
    ])

# Login Page
if st.session_state.selected_tab == "Login" and not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Login to ResumeSmartX</h1>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if login_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.selected_tab = "🏆 Resume Analysis"
                st.success("Logged in successfully!")
                st.rerun()

# Registration Page
elif st.session_state.selected_tab == "Register" and not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Register for ResumeSmartX</h1>", unsafe_allow_html=True)
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match.")
            elif len(username) < 3 or len(password) < 6:
                st.error("Username must be at least 3 characters and password at least 6 characters.")
            else:
                if register_user(username, password):
                    st.success("Registration successful! Please log in.")
                    st.session_state.selected_tab = "Login"
                    st.rerun()

# Main App
elif st.session_state.authenticated:
    st.sidebar.write(f"Welcome, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.selected_tab = "Login"
        st.rerun()

    if st.session_state.selected_tab == "🏆 Resume Analysis":
        st.markdown("""
            <h1 style='text-align: center; color: #4CAF50;'>MY PERSONAL ATS</h1>
            <hr style='border: 1px solid #4CAF50;'>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            input_text = st.text_area("📋 Job Description:", key="input", height=150)
        uploaded_file = None
        resume_text = ""
        with col2:
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
                    save_resume_to_postgres(uploaded_file.name, resume_text)
                except Exception as e:
                    st.error(f"❌ Failed to read PDF: {str(e)}")
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🛠 Quick Actions</h3>", unsafe_allow_html=True)
        response_container = st.container()

        if st.button("📖 Tell Me About the Resume"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    prompt_text = f"Please review the following resume and provide a detailed evaluation: {resume_text}"
                    response = get_gemini_response(prompt_text, action="Tell_me_about_resume")
                    log_to_postgres("Tell_me_about_resume", response)
                    st.session_state['resume_response'] = response
                    st.write(response)
                    st.download_button("💾 Download Resume Evaluation", response, "resume_evaluation.txt")
                    if st.button("🔊 Read Resume Summary"):
                        with st.spinner("🎤 Generating audio..."):
                            try:
                                short_text = response[:2000]
                                audio = client.generate(
                                    text=short_text,
                                    voice="Rachel",
                                    model="eleven_multilingual-v2"
                                )
                                with open("resume_summary.mp3", "wb") as f:
                                    f.write(audio)
                                st.success("✅ Audio summary ready!")
                                st.audio("resume_summary.mp3")
                            except Exception as e:
                                st.error(f"❌ Error generating audio: {str(e)}")
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("📊 Percentage Match"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text and input_text:
                    response = get_gemini_response(
                        f"Evaluate the following resume against this job description and provide a percentage match in first :\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}",
                        action="Percentage_Match"
                    )
                    log_to_postgres("Percentage_Match", response)
                    st.write(response)
                    st.download_button("💾 Download Percentage Match", response, "percentage_match.txt")
                else:
                    st.warning("⚠ Please upload a resume and provide a job description.")

        learning_path_duration = st.selectbox("📆 Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])
        if st.button("🎓 Personalized Learning Path"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text and input_text and learning_path_duration:
                    response = get_gemini_response(
                        f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text} and also suggest books and other important thing",
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
                else:
                    st.warning("⚠ Please upload a resume and provide a job description.")

        if st.button("📝 Generate Updated Resume"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(
                        f"Suggest improvements and generate an updated resume for this candidate according to job description, not more than 2 pages:\n{resume_text}",
                        action="Generate_Updated_Resume"
                    )
                    log_to_postgres("Generate_Updated_Resume", response)
                    st.write(response)
                    pdf_file = "updated_resume.pdf"
                    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = [Paragraph(response.replace('\n', '<br/>'), styles['Normal'])]
                    doc.build(story)
                    with open(pdf_file, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="📥 Download Updated Resume", 
                        data=pdf_data, 
                        file_name="Updated_Resume.pdf", 
                        mime="application/pdf"
                    )
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("❓ Generate 30 Interview Questions and Answers"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(
                        "Generate 30 technical interview questions and their detailed answers according to that job description.",
                        action="Generate_Interview_Questions"
                    )
                    log_to_postgres("Generate_Interview_Questions", response)
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("🚀 Skill Development Plan"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text and input_text:
                    response = get_gemini_response(
                        f"Based on the resume and job description, suggest courses, books, and projects to improve the candidate's weak or missing skills.\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}",
                        action="Skill_Development_Plan"
                    )
                    log_to_postgres("Skill_Development_Plan", response)
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("🎥 Mock Interview Questions"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text and input_text:
                    response = get_gemini_response(
                        f"Generate follow-up interview questions based on the resume and job description, simulating a live interview.\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}",
                        action="Mock_Interview_Questions"
                    )
                    log_to_postgres("Mock_Interview_Questions", response)
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("💡 AI-Driven Insights"):
            with st.spinner("🔍 Analyzing... Please wait"):
                if resume_text:
                    recommendations = get_gemini_response(
                        f"Based on this resume, suggest specific job roles the user is most suited for and analyze market trends for their skills.\n\nResume:\n{resume_text}",
                        action="AI_Driven_Insights"
                    )
                    log_to_postgres("AI_Driven_Insights", recommendations)
                    try:
                        recommendations = json.loads(recommendations)
                        st.write("📋 Smart Recommendations:")
                        st.write(recommendations.get("job_roles", "No recommendations found."))
                        st.write("📊 Market Trends:")
                        st.write(recommendations.get("market_trends", "No market trends available."))
                    except json.JSONDecodeError:
                        st.write("📋 AI-Driven Insights:")
                        st.write(recommendations)
                else:
                    st.warning("⚠ Please upload a resume first.")

    elif st.session_state.selected_tab == "🔝Top 3 MNCs":
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color:#FFA500;'>🚀 MNC Data Science Preparation</h2>", unsafe_allow_html=True)
        st.markdown("---")
        if "selected_mnc" not in st.session_state:
            st.session_state["selected_mnc"] = None
        mnc_data = [
            {"name": "TCS", "color": "#FFA500", "icon": "🎯"},
            {"name": "Infosys", "color": "#03A9F4", "icon": "🚀"},
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
            with st.spinner("⏳ Analyzing your resume... Please wait"):
                if "resume_text" in st.session_state and st.session_state["resume_text"]:
                    resume_text = st.session_state["resume_text"]
                    response = get_gemini_response(
                        f"Based on the candidate's qualifications and resume, what additional skills and knowledge are needed to secure a Data Science role at {selected_mnc}?",
                        action="Additional_Skills_MNCS"
                    )
                    log_to_postgres("Additional_Skills_MNCS", response)
                    st.info(response)
                else:
                    st.warning("⚠ Please upload a resume first.")
            if st.button("📂 Project Types & Required Skills"):
                with st.spinner("⏳ Loading... Please wait"):
                    if "resume_text" in st.session_state and st.session_state["resume_text"]:
                        resume_text = st.session_state["resume_text"]
                        response = get_gemini_response(
                            f"What types of Data Science projects does {selected_mnc} typically work on, and what skills align best?",
                            action="Project_Types_Skills"
                        )
                        log_to_postgres("Project_Types_Skills", response)
                        st.success(response)
                    else:
                        st.warning("⚠ Please upload a resume first.")
            if st.button("🛠 Required Skills"):
                with st.spinner("⏳ Loading... Please wait"):
                    if "resume_text" in st.session_state and st.session_state["resume_text"]:
                        resume_text = st.session_state["resume_text"]
                        response = get_gemini_response(
                            f"What key technical and soft skills are needed for a Data Science role at {selected_mnc}?",
                            action="Required_Skills"
                        )
                        log_to_postgres("Required_Skills", response)
                        st.success(response)
                    else:
                        st.warning("⚠ Please upload a resume first.")
            if st.button("💡 Career Recommendations"):
                with st.spinner("⏳ Loading... Please wait"):
                    if "resume_text" in st.session_state and st.session_state["resume_text"]:
                        resume_text = st.session_state["resume_text"]
                        response = get_gemini_response(
                            f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at {selected_mnc}?",
                            action="Career_Recommendations"
                        )
                        log_to_postgres("Career_Recommendations", response)
                        st.success(response)
                    else:
                        st.warning("⚠ Please upload a resume first.")

    elif st.session_state.selected_tab == "📊 DSA & Data Science":
        st.markdown("<h3 style='text-align: center;'>🛠 DSA for Data Science</h3>", unsafe_allow_html=True)
        level = st.selectbox("📚 Select Difficulty Level:", ["Easy", "Intermediate", "Advanced"])
        if st.button(f"📝 Generate {level} DSA Questions (Data Science)"):
            with st.spinner("⏳ Loading... Please wait"):
                response = get_gemini_response(
                    f"Generate 10 DSA questions and answers for data science at {level} level.",
                    action="DSA_Questions"
                )
                log_to_postgres("DSA_Questions", response)
                st.write(response)
        topic = st.selectbox("🗂 Select DSA Topic:", [
            "Arrays", "Linked Lists", "Trees", "Graphs", "Dynamic Programming", 
            "Recursion", "algorithm complexity (Big O notation)", "sorting", "searching"
        ])
        if st.button(f"📖 Teach me {topic} with Case Studies"):
            with st.spinner("⏳ Gathering resources... Please wait"):
                explanation_response = get_gemini_response(
                    f"Explain the {topic} topic in an easy-to-understand way suitable for beginners, using simple language and clear examples add all details like defination exampales of {topic} and code implementation in python with full explaination of that code.",
                    action="Teach_me_DSA_Topics"
                )
                log_to_postgres("Teach_me_DSA_Topics", explanation_response)
                st.write(explanation_response)
                case_study_response = get_gemini_response(
                    f"Provide a real-world case study on {topic} for data science/ data engineer/ m.l/ai with a detailed, easy-to-understand solution.",
                    action="Case_Study_DSA_Topics"
                )
                log_to_postgres("Case_Study_DSA_Topics", case_study_response)
                st.write(case_study_response)

    elif st.session_state.selected_tab == "📚 Question Bank":
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color:#FFA500;'>📚 Question Bank</h2>", unsafe_allow_html=True)
        st.markdown("---")
        question_category = st.selectbox("❓ Select Question Category:", [
            "Python", "Machine Learning", "Deep Learning", "Docker", 
            "Data Warehousing", "Data Pipelines", "Data Modeling", "SQL"
        ])
        if st.button(f"📝 Generate 30 {question_category} Interview Questions"):
            with st.spinner("⏳ Loading... Please wait"):
                response = get_gemini_response(
                    f"Generate 30 {question_category} interview questions and detailed answers",
                    action="Interview_Questions"
                )
                log_to_postgres("Interview_Questions", response)
                st.write(response)

    elif st.session_state.selected_tab == "🛠️ Code Debugger":
        st.markdown("<h3 style='text-align: center;'>🛠️ Python Code Debugger</h3>", unsafe_allow_html=True)
        user_code = st.text_area("Paste your Python code below:", height=300)
        if st.button("Check & Fix Code"):
            if user_code.strip() == "":
                st.warning("Please enter some code.")
            else:
                with st.spinner("Analyzing and fixing code..."):
                    prompt = f"""
                    Analyze the following Python code for bugs, syntax errors, and logic errors.
                    If it has issues, correct them. Return the fixed code and briefly explain the changes made.
                    Code:
                    ```python
                    {user_code}
                    ```
                    """
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content([prompt])
                        if response:
                            st.subheader("✅ Corrected Code")
                            st.code(response.text, language="python")
                            log_to_postgres("Corrected Code", response.text)
                        else:
                            st.error("No response from Gemini.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif st.session_state.selected_tab == "🤖 Voice Agent Chat":
        st.markdown("""
            <h1 style='text-align: center; color: #4CAF50;'>Talk to AI Interviewer 🤖🎤</h1>
            <hr style='border: 1px solid #4CAF50;'>
            <p style='text-align: center;'>Start a real-time voice conversation with our AI agent powered by ElevenLabs.</p>
            <div style='text-align: center; margin-bottom: 30px;'>
                <a href='https://elevenlabs.io/app/talk-to?agent_id=ybbzwh5ejKaruGyPH3pg' target='_blank'>
                    <button style='padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer;'>
                        🚀 Launch Voice Interview Agent
                    </button>
                </a>
            </div>
        """, unsafe_allow_html=True)

    # Custom CSS
    custom_css = """
    <style>
        .bottom-right {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 14px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease-in-out;
        }
        .bottom-right:hover {
            transform: scale(1.1);
        }
    </style>
    <div class="bottom-right"> <b>Built by AI Team of Regex Software </b></div>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

else:
    st.error("Please log in to access the application.")