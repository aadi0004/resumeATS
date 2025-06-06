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

from dotenv import load_dotenv
import os

load_dotenv()
# Get API credentials from environment
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")


# -------------------- ✅ LOGGING SETUP START --------------------
import os
import csv
import threading
from datetime import datetime

# Logging setup
csv_lock = threading.Lock()

LOG_DIR = ".logs"
LOG_FILE = os.path.join(LOG_DIR, "api_usage_logs.csv")

# Create .logs folder and CSV file if not exist
os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Action", "API_Hits", "Tokens_Generated", "Total_Tokens_Till_Now"])


# Helper to calculate current total tokens from the CSV file
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


# Logging function
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
# PostgreSQL DB Logging
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS button_logs (
                id SERIAL PRIMARY KEY,
                action VARCHAR(255),
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO button_logs (action, response)
            VALUES (%s, %s)
        """, (action, response))
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255),
                resume_text TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO resumes (filename, resume_text)
            VALUES (%s, %s)
        """, (filename, resume_text))
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
            token_count = len(prompt.split())  # Estimate token count
            log_api_usage(action, token_count)
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        log_api_usage(f"{action}_Error", 0)
        return f"API Error: {str(e)}"
# ----------------------------------------------------------------

st.set_page_config(page_title="ResumeSmartX - AI ATS", page_icon="📄", layout='wide')
# Sidebar Navigation
st.sidebar.image("logo.png", width=200)
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Choose a Feature", [
    "🏆 Resume Analysis", "📚 Question Bank", "📊 DSA & Data Science","🔝Top 3 MNCs","🗣️ Group discussion","🛠️ Code Debugger","🧠 Mock Interview","🤖 Voice Agent Chat"
])

if selected_tab == "🏆 Resume Analysis":

    # Header with a fresh style
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>MY PERSONAL ATS</h1>
        <hr style='border: 1px solid #4CAF50;'>
    """, unsafe_allow_html=True)

    # Input section with better layout
    col1, col2 = st.columns(2)

    with col1:
        input_text = st.text_area("📋 Job Description:", key="input", height=150)

    import streamlit as st
    from PyPDF2 import PdfReader

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
                # Store in session state
                st.session_state['resume_text'] = resume_text
                # Save to PostgreSQL
                save_resume_to_postgres(uploaded_file.name, resume_text)
            except Exception as e:
                st.error(f"❌ Failed to read PDF: {str(e)}")

    # Always visible buttons styled
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>🛠 Quick Actions</h3>", unsafe_allow_html=True)

    # Full-width response area
    response_container = st.container()


    import streamlit as st
    from elevenlabs.client import ElevenLabs
    from dotenv import load_dotenv
    import os

    # Load .env
    load_dotenv()
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

    if st.button("📖 Tell Me About the Resume"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text:
                prompt_text = f"Please review the following resume and provide a detailed evaluation: {resume_text}"
                
                response = get_gemini_response(prompt_text, action="Tell_me_about_resume")
                log_to_postgres("Tell_me_about_resume", response)
                st.write(response)

                st.download_button("💾 Download Resume Evaluation", response, "resume_evaluation.txt")

                # 🔊 Add ElevenLabs audio narration button
                if "resume_response" not in st.session_state:
                    if st.button("🔊 Read Resume Summary"):
                        with st.spinner("🎤 Generating audio..."):
                            try:
                                short_text = st.session_state['resume_response'][:2000]  # Limit to 2000 chars
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

  
    if st.button("📊 Percentage Match"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text and input_text:
                response = get_gemini_response(f"Evaluate the following resume against this job description and provide a percentage match in first :\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}",
                                                action="Percentage_Match")
                log_to_postgres("Percentage_Match", response)
                st.write(response)
                st.download_button("💾 Download Percentage Match", response, "percentage_match.txt")
            else:
                st.warning("⚠ Please upload a resume and provide a job description.")

    learning_path_duration = st.selectbox("📆 Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])
    if st.button("🎓 Personalized Learning Path"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text and input_text and learning_path_duration:
                response = get_gemini_response(f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text} and also suggest books and other important thing",
                                                action="Personalized_Learning_Path")
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
                st.download_button(f"💾 Download Learning Path PDF", pdf_buffer.getvalue(), f"learning_path_{learning_path_duration.replace(' ', '_').lower()}.pdf", "application/pdf")
            else:
                st.warning("⚠ Please upload a resume and provide a job description.")

    if st.button("📝 Generate Updated Resume"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text:
                response = get_gemini_response(f"Suggest improvements and generate an updated resume for this candidate according to job description, not more than 2 pages:\n{resume_text}",
                                                action="Generate_Updated_Resume")
                log_to_postgres("Generate_Updated_Resume", response)
                st.write(response)

                # Convert response to PDF
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph
                from reportlab.lib.styles import getSampleStyleSheet

                pdf_file = "updated_resume.pdf"
                doc = SimpleDocTemplate(pdf_file, pagesize=letter)
                styles = getSampleStyleSheet()
                story = [Paragraph(response.replace('\n', '<br/>'), styles['Normal'])]
                doc.build(story)

                # Read PDF as binary
                with open(pdf_file, "rb") as f:
                    pdf_data = f.read()

                # Download button for PDF
                st.download_button(label="📥 Download Updated Resume", data=pdf_data, file_name="Updated_Resume.pdf", mime="application/pdf")
            else:
                st.warning("⚠ Please upload a resume first.")


    if st.button("❓ Generate 30 Interview Questions and Answers"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text:
                response = get_gemini_response("Generate 30 technical interview questions and their detailed answers according to that job description.",
                                                action="Generate_Interview_Questions")
                log_to_postgres("Generate_Interview_Questions", response)
                st.write(response)
            else:
                st.warning("⚠ Please upload a resume first.")


    if st.button("🚀 Skill Development Plan"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text and input_text:
                response = get_gemini_response(f"Based on the resume and job description, suggest courses, books, and projects to improve the candidate's weak or missing skills.\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}",
                                                action="Skill_Development_Plan")
                log_to_postgres("Skill_Development_Plan", response)
                st.write(response)
            else:
                st.warning("⚠ Please upload a resume first.")

    if st.button("🎥 Mock Interview Questions"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text and input_text:
                response = get_gemini_response(f"Generate follow-up interview questions based on the resume and job description, simulating a live interview.\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}",
                                                action="Mock_Interview_Questions")
                log_to_postgres("Mock_Interview_Questions", response)
                st.write(response)
            else:
                st.warning("⚠ Please upload a resume first.")

    import json

    if st.button("💡 AI-Driven Insights"):
        with st.spinner("🔍 Analyzing... Please wait"):
            if resume_text:
                recommendations = get_gemini_response(f"Based on this resume, suggest specific job roles the user is most suited for and analyze market trends for their skills.\n\nResume:\n{resume_text}",
                                                    action="AI_Driven_Insights")
                log_to_postgres("AI_Driven_Insights", response)
                try:
                    recommendations = json.loads(recommendations)  # Attempt to parse JSON
                    st.write("📋 Smart Recommendations:")
                    st.write(recommendations.get("job_roles", "No recommendations found."))
                    st.write("📊 Market Trends:")
                    st.write(recommendations.get("market_trends", "No market trends available."))
                except json.JSONDecodeError:
                    # Fallback if response is not JSON
                    st.write("📋 AI-Driven Insights:")
                    st.write(recommendations)
            else:
                st.warning("⚠ Please upload a resume first.")


elif selected_tab == "🔝Top 3 MNCs":

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


# st.markdown("---")

elif selected_tab == "📊 DSA & Data Science":

    # st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>🛠 DSA for Data Science</h3>", unsafe_allow_html=True)

    # Main DSA Questions button
    level = st.selectbox("📚 Select Difficulty Level:", ["Easy", "Intermediate", "Advanced"])

    if st.button(f"📝 Generate {level} DSA Questions (Data Science)"):
        with st.spinner("⏳ Loading... Please wait"):
            response = get_gemini_response(f"Generate 10 DSA questions and answers for data science at {level} level.",
                                            action="DSA_Questions")
            log_to_postgres("DSA_Questions", response)
            st.write(response)


    topic = st.selectbox("🗂 Select DSA Topic:", ["Arrays", "Linked Lists", "Trees", "Graphs", "Dynamic Programming", "Recursion","algorithm complexity (Big O notation)","sorting" , "searching"])

    if st.button(f"📖 Teach me {topic} with Case Studies"):
        with st.spinner("⏳ Gathering resources... Please wait"):
            explanation_response = get_gemini_response(f"Explain the {topic} topic in an easy-to-understand way suitable for beginners, using simple language and clear examples add all details like defination exampales of {topic} and code implementation in python with full explaination of that code.",
                                                        action="Teach_me_DSA_Topics")
            log_to_postgres("Teach_me_DSA_Topics", explanation_response)
            st.write(explanation_response)

            case_study_response = get_gemini_response(f"Provide a real-world case study on {topic} for data science/ data engineer/ m.l/ai with a detailed, easy-to-understand solution.",
                                                    action="Case_Study_DSA_Topics")
            log_to_postgres("Case_Study_DSA_Topics", case_study_response)
            st.write(case_study_response)



    st.markdown("---")

elif selected_tab == "📚 Question Bank":
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color:#FFA500;'>📚 Question Bank</h2>", unsafe_allow_html=True)
    st.markdown("---")
    question_category = st.selectbox("❓ Select Question Category:", ["Python", "Machine Learning", "Deep Learning", "Docker", "Data Warehousing", "Data Pipelines", "Data Modeling", "SQL"])

    if st.button(f"📝 Generate 30 {question_category} Interview Questions"):
        with st.spinner("⏳ Loading... Please wait"):
            response = get_gemini_response(f"Generate 30 {question_category} interview questions and detailed answers",
                                            action="Interview_Questions")
            log_to_postgres("Interview_Questions", response)
            st.write(response)
            st.write(response)












if selected_tab == "🗣️ Group discussion":

    import streamlit as st

    # Mock AI response function (Replace this with your actual API call)
    def get_gemini_response(prompt, action="Gemini"):
        topic_map = {
            "Data Science": [
                "What is the role of data science in modern industries?",
                "Explain the difference between supervised and unsupervised learning.",
                "How do data cleaning techniques impact model performance?",
                "What are the ethical concerns in data science?",
                "How can data science be used for real-world problem-solving?"
            ],
            "AI": [
                "What is Artificial Intelligence, and how does it work?",
                "Discuss the difference between AI, Machine Learning, and Deep Learning.",
                "What are the main applications of AI in daily life?",
                "What are the ethical risks associated with AI development?",
                "What is the future of AI in automation and job markets?"
            ],
            "Machine Learning": [
                "Define Machine Learning and its core principles.",
                "How does feature selection impact model accuracy?",
                "What is the bias-variance tradeoff in ML?",
                "Explain reinforcement learning with an example.",
                "How do you handle overfitting in machine learning?"
            ],
            "Web Development": [
                "What are the key components of web development?",
                "Explain the difference between frontend and backend development.",
                "How does responsive design impact user experience?",
                "What are the security best practices for web applications?",
                "What is the role of APIs in modern web applications?"
            ]
        }
        return topic_map.get(prompt, ["No questions found."])

    def ai_guided_discussion():
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🤖 AI-Guided Group Discussion</h3>", unsafe_allow_html=True)

        topics = ["Data Science", "AI", "Machine Learning", "Web Development"]
        selected_topic = st.selectbox("📌 Select Discussion Topic:", topics)

        # Initialize session state
        if 'selected_topic' not in st.session_state or st.session_state.selected_topic != selected_topic:
            st.session_state.selected_topic = selected_topic
            st.session_state.question_index = 0
            st.session_state.questions = get_gemini_response(selected_topic, action="AI_Guided_Discussion")
            st.session_state.answers = []
            st.session_state.feedback = []

        questions = st.session_state.questions

        if st.session_state.question_index < len(questions):
            current_question = questions[st.session_state.question_index]
            st.markdown(f"**🤖 AI:** {current_question}")
            
            user_response = st.text_area("✍️ Your Answer:", key=f"answer_{st.session_state.question_index}")

            if st.button("Submit Answer"):
                if not user_response.strip():
                    st.warning("⚠️ Please enter a response before submitting.")
                else:
                    st.session_state.answers.append(user_response)
                    
                    # Get AI feedback (Mocked)
                    feedback = f"Good response! You covered {selected_topic} well."
                    st.session_state.feedback.append(feedback)
                    
                    st.markdown(f"**✅ AI Feedback:** {feedback}")

                    # Move to next question
                    st.session_state.question_index += 1
                    st.rerun()

        else:
            st.success("🎉 Discussion completed! Here’s a summary:")
            for i, (q, ans, fb) in enumerate(zip(st.session_state.questions, st.session_state.answers, st.session_state.feedback)):
                st.markdown(f"**🔹 Q{i+1}:** {q}")
                st.markdown(f"💡 **Your Answer:** {ans}")
                st.markdown(f"✅ **AI Feedback:** {fb}")
                st.markdown("---")

            if st.button("Restart Discussion"):
                st.session_state.clear()
                st.rerun()

    # Run the function
    ai_guided_discussion()


# st.markdown("---")
elif selected_tab == "🛠️ Code Debugger":

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
                        response= st.code(response.text, language="python")
                        log_to_postgres("Corrected Code", response)
                    else:
                        st.error("No response from Gemini.")
                except Exception as e:
                    st.error(f"Error: {e}")








# Set your ElevenLabs API Key from .env

# --- MOCK INTERVIEW TAB ---
   

# Set your ElevenLabs API Key from .env


# Get API credentials from environment

# ------------------------ Streamlit Page ------------------------
elif selected_tab == "🤖 Voice Agent Chat":
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



# Custom CSS for bottom-right placement and pop-up effect
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

# Inject CSS into Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)