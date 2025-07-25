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
from datetime import datetime, timedelta
import json
import streamlit.components.v1 as components
import requests
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
import csv
import threading
import logging
from JobSearchClient import JobSearchClient

# Set up logging for debugging LangGraph
logging.basicConfig(level=logging.DEBUG, filename="langgraph_debug.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
logging.debug("Environment variables loaded")
logging.debug(f"TAVILY_API_KEY: {os.getenv('TAVILY_API_KEY')[:5]}...")  # Log partial key for security
logging.debug(f"SERPER_API_KEY: {os.getenv('SERPER_API_KEY')[:5]}...")  # Log partial key for security
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# -------------------- ✅ LOGGING SETUP START --------------------
LOG_DIR = ".logs"
LOG_FILE = os.path.join(LOG_DIR, "api_usage_logs.csv")
csv_lock = threading.Lock()

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

# -------------------- ✅ Gemini API Wrapper --------------------
def get_gemini_response(prompt, action="Gemini_API_Call"):
    if not prompt.strip():
        logging.error("Empty prompt provided to Gemini API")
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
            logging.info(f"Gemini API call successful for action: {action}, tokens: {token_count}")
            return response.text
        else:
            logging.error("No valid response from Gemini API")
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        log_api_usage(f"{action}_Error", 0)
        logging.error(f"Gemini API error: {str(e)}")
        return f"API Error: {str(e)}"
# ----------------------------------------------------------------

# -------------------- ✅ LangGraph for Job Search --------------------
class JobSearchState:
    def __init__(self, resume_text: str = "", job_field: str = "", skills: List[str] = [], job_listings: List[Dict[str, Any]] = []):
        self.resume_text = resume_text
        self.job_field = job_field
        self.skills = skills
        self.job_listings = job_listings

def extract_skills(state: Dict[str, Any]) -> Dict[str, Any]:
    logging.debug(f"Extracting skills from state: {state}")
    skills = []
    if state.get("resume_text"):
        prompt = f"Extract key skills from the following resume as a concise comma-separated list (e.g., Python, SQL, Machine Learning, AWS):\n\n{state['resume_text']}"
        skills_response = get_gemini_response(prompt, action="Extract_Skills")
        try:
            # Attempt to parse as JSON if the response is structured
            if "{" in skills_response:
                skills = json.loads(skills_response).get("skills", [])
            else:
                # Clean and split the response into a list
                skills = [skill.strip() for skill in skills_response.split(",") if skill.strip()]
        except json.JSONDecodeError:
            # Fallback to splitting by commas and cleaning
            skills = [skill.strip() for skill in skills_response.split(",") if skill.strip()]
    else:
        skills = state.get("job_field", "").split() if state.get("job_field") else []
    logging.debug(f"Extracted skills: {skills}")
    return {"skills": skills}

def search_jobs(state: Dict[str, Any]) -> Dict[str, Any]:
    logging.debug(f"Searching jobs with state: {state}")
    skills = state.get("skills", [])
    job_field = state.get("job_field", "")
    query = ", ".join(skills) if skills else job_field
    if not query:
        logging.error("No skills or job field provided for job search")
        return {"job_listings": [{"error": "No skills or job field provided. Please enter a job field or upload a resume."}]}
    
    # Initialize JobSearchClient
    job_search_client = JobSearchClient(tavily_api_key=TAVILY_API_KEY, serper_api_key=SERPER_API_KEY)
    job_listings = job_search_client.search_jobs(query=query, max_results=10)
    
    # Fallback to job_field if no results and job_field exists
    if not job_listings or "error" in job_listings[0]:
        if job_field and job_field != query:
            logging.debug(f"Fallback to job_field query: {job_field}")
            job_listings = job_search_client.search_jobs(query=job_field, max_results=10)
    
    logging.debug(f"Final job listings: {job_listings}")
    return {"job_listings": job_listings}

def format_jobs(state: Dict[str, Any]) -> Dict[str, Any]:
    logging.debug(f"Formatting jobs with state: {state}")
    job_listings = state.get("job_listings", [])
    if not job_listings or "error" in job_listings[0]:
        logging.warning("No valid job listings to format")
        return {"job_listings": job_listings}
    formatted_jobs = []
    for job in job_listings:
        formatted_jobs.append({
            "title": job.get("title", "N/A"),
            "company": job.get("company", "N/A"),
            "location": job.get("location", "India"),
            "apply_link": f"<a href='{job.get('link', '#')}' target='_blank'>Apply Here</a>"
        })
    logging.debug(f"Formatted jobs: {formatted_jobs}")
    return {"job_listings": formatted_jobs}

workflow = StateGraph(dict)
workflow.add_node("extract_skills", extract_skills)
workflow.add_node("search_jobs", search_jobs)
workflow.add_node("format_jobs", format_jobs)
workflow.set_entry_point("extract_skills")
workflow.add_edge("extract_skills", "search_jobs")
workflow.add_edge("search_jobs", "format_jobs")
workflow.add_edge("format_jobs", END)
job_search_graph = workflow.compile()
# ----------------------------------------------------------------

st.set_page_config(page_title="ResumeSmartX - AI ATS", page_icon="📄", layout='wide')

# Sidebar Navigation
st.sidebar.image("logo.png", width=200)
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Choose a Feature", [
    "🏆 Resume Analysis", "📚 Question Bank", "📊 DSA & Data Science", "🔝 Top 3 MNCs",
    "🗣️ Group Discussion", "🛠️ Code Debugger", "🧠 Mock Interview", "🤖 Voice Agent Chat",
    "🔍 Fetch Recent Jobs in India"
])

# --- RESUME ANALYSIS TAB ---
if selected_tab == "🏆 Resume Analysis":
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>MY PERSONAL ATS</h1>
        <hr style='border: 1px solid #4CAF50;'>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        input_text = st.text_area("📋 Job Description:", key="input", height=150)
    with col2:
        uploaded_file = st.file_uploader("📄 Upload your resume (PDF)...", type=['pdf'])
        resume_text = ""
        if uploaded_file:
            st.success("✅ PDF Uploaded Successfully.")
            try:
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    if page and page.extract_text():
                        resume_text += page.extract_text()
                st.session_state['resume_text'] = resume_text
            except Exception as e:
                st.error(f"❌ Failed to read PDF: {str(e)}")
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>🛠 Quick Actions</h3>", unsafe_allow_html=True)
    response_container = st.container()

    if st.button("📖 Tell Me About the Resume"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text:
                response = get_gemini_response(
                    f"Please review the following resume and provide a detailed evaluation: {resume_text}",
                    action="Tell_me_about_resume"
                )
                st.write(response)
                st.download_button("💾 Download Resume Evaluation", response, "resume_evaluation.txt")
            else:
                st.warning("⚠ Please upload a resume first.")

    if st.button("📊 Percentage Match"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text and input_text:
                response = get_gemini_response(
                    f"Evaluate the following resume against this job description and provide a percentage match first:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}",
                    action="Percentage_Match"
                )
                st.write(response)
                st.download_button("💾 Download Percentage Match", response, "percentage_match.txt")
            else:
                st.warning("⚠ Please upload a resume and provide a job description.")

    learning_path_duration = st.selectbox("📆 Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])
    if st.button("🎓 Personalized Learning Path"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text and input_text and learning_path_duration:
                response = get_gemini_response(
                    f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text} and also suggest books and other important things",
                    action="Personalized_Learning_Path"
                )
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

# --- QUESTION BANK TAB ---
elif selected_tab == "📚 Question Bank":
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
            st.write(response)

# --- DSA & DATA SCIENCE TAB ---
elif selected_tab == "📊 DSA & Data Science":
    st.markdown("<h3 style='text-align: center;'>🛠 DSA for Data Science</h3>", unsafe_allow_html=True)
    level = st.selectbox("📚 Select Difficulty Level:", ["Easy", "Intermediate", "Advanced"])
    if st.button(f"📝 Generate {level} DSA Questions (Data Science)"):
        with st.spinner("⏳ Loading... Please wait"):
            response = get_gemini_response(
                f"Generate 10 DSA questions and answers for data science at {level} level.",
                action="DSA_Questions"
            )
            st.write(response)
    topic = st.selectbox("🗂 Select DSA Topic:", [
        "Arrays", "Linked Lists", "Trees", "Graphs", "Dynamic Programming",
        "Recursion", "Algorithm Complexity (Big O Notation)", "Sorting", "Searching"
    ])
    if st.button(f"📖 Teach me {topic} with Case Studies"):
        with st.spinner("⏳ Gathering resources... Please wait"):
            explanation_response = get_gemini_response(
                f"Explain the {topic} topic in an easy-to-understand way suitable for beginners, using simple language and clear examples add all details like definition, examples of {topic}, and code implementation in python with full explanation of that code.",
                action="Teach_me_DSA_Topics"
            )
            st.write(explanation_response)
            case_study_response = get_gemini_response(
                f"Provide a real-world case study on {topic} for data science/data engineer/ML/AI with a detailed, easy-to-understand solution.",
                action="Case_Study_DSA_Topics"
            )
            st.write(case_study_response)

# --- TOP 3 MNCs TAB ---
elif selected_tab == "🔝 Top 3 MNCs":
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
                    st.success(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

# --- GROUP DISCUSSION TAB ---
elif selected_tab == "🗣️ Group Discussion":
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
                    feedback = f"Good response! You covered {selected_topic} well."
                    st.session_state.feedback.append(feedback)
                    st.markdown(f"**✅ AI Feedback:** {feedback}")
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
    ai_guided_discussion()

# --- CODE DEBUGGER TAB ---
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
                    else:
                        st.error("No response from Gemini.")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- MOCK INTERVIEW TAB ---
elif selected_tab == "🧠 Mock Interview":
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>Mock Interview Assistant 🎙️</h1>
        <hr style='border: 1px solid #4CAF50;'>
    """, unsafe_allow_html=True)
    st.markdown("Upload your <b>resume</b> and <b>job description</b> to begin the mock interview.", unsafe_allow_html=True)
    resume_file = st.file_uploader("📄 Upload Resume (PDF/DOCX)", type=['pdf', 'docx'], key="resume_file")
    jd_file = st.file_uploader("📝 Upload Job Description (Text/PDF/DOCX)", type=['txt', 'pdf', 'docx'], key="jd_file")
    def extract_text(file):
        text = ""
        if file.type == "application/pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            text = docx2txt.process(path)
        elif file.type == "text/plain":
            text = str(file.read(), 'utf-8')
        return text
    def generate_voice(text, voice_id="Rachel"):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            audio_path = os.path.join(tempfile.gettempdir(), f"question_{text[:10]}.mp3")
            with open(audio_path, "wb") as f:
                f.write(response.content)
            return audio_path
        else:
            st.error("🚫 Failed to generate voice from ElevenLabs API.")
            return None
    def generate_questions(resume_text, jd_text):
        return [
            "Tell me about yourself.",
            "Can you explain a recent project related to the job role?",
            "What are your strengths and weaknesses?",
            "Why do you want this job?",
            "Tell me about a time you handled a challenge."
        ]
    def evaluate_response(transcript, jd_text):
        if any(keyword in transcript.lower() for keyword in jd_text.lower().split()[:10]):
            return "✅ Good match! You covered relevant points."
        return "⚠️ Try to include more role-specific keywords."
    if resume_file and jd_file:
        resume_text = extract_text(resume_file)
        jd_text = extract_text(jd_file)
        questions = generate_questions(resume_text, jd_text)
        st.success("✅ Documents processed. Starting the mock interview...")
        for i, question in enumerate(questions):
            st.markdown(f"### Question {i+1}:")
            st.markdown(f"**{question}**")
            audio_path = generate_voice(question)
            if audio_path:
                st.audio(audio_path, format='audio/mp3')
            st.markdown("**🎤 Record your response using a voice recorder and upload it below.**")
            audio_response = st.file_uploader(f"Upload your voice response to Question {i+1}", type=['wav', 'mp3'], key=f"audio_response_{i}")
            if audio_response:
                st.audio(audio_response, format='audio/mp3')
                transcript = st.text_area(f"📝 Transcribe your voice response for feedback (manual input):", key=f"transcript_{i}")
                if transcript:
                    feedback = evaluate_response(transcript, jd_text)
                    st.markdown(f"**📜 Feedback:** {feedback}")
        st.markdown("---")
        st.success("🎉 Mock interview complete! Review your answers and feedback above.")
    else:
        st.info("ℹ️ Please upload both resume and job description to begin.")

# --- VOICE AGENT CHAT TAB ---
elif selected_tab == "🤖 Voice Agent Chat":
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>Talk to AI Interviewer 🤖🎤</h1>
        <hr style='border: 1px solid #4CAF50;'>
        <p style='text-align: center;'>Start a real-time voice conversation with our AI agent powered by ElevenLabs.</p>
        <div style='text-align: center; margin-bottom: 30px;'>
            <a href='https://elevenlabs.io/app/talk-to?agent_id=Sy2RXopFB3RH3mhEicI3' target='_blank'>
                <button style='padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer;'>
                    🚀 Launch Voice Interview Agent
                </button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# --- FETCH RECENT JOBS IN INDIA TAB ---
elif selected_tab == "🔍 Fetch Recent Jobs in India":
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>🔍 Recent Job Openings in India</h1>
        <hr style='border: 1px solid #4CAF50;'>
    """, unsafe_allow_html=True)
    st.markdown("Enter a job field or use the resume uploaded in the Resume Analysis tab to find the latest job postings in India.")
    resume_text = st.session_state.get('resume_text', '')
    if resume_text:
        st.success("✅ Using resume from Resume Analysis tab.")
    else:
        resume_file = st.file_uploader("📄 Upload Resume (PDF/DOCX) for Tailored Jobs", type=['pdf', 'docx'], key="job_resume_file")
        if resume_file:
            if resume_file.type == "application/pdf":
                reader = PdfReader(resume_file)
                for page in reader.pages:
                    resume_text += page.extract_text() or ""
            elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, resume_file.name)
                with open(path, "wb") as f:
                    f.write(resume_file.getbuffer())
                resume_text = docx2txt.process(path)
            st.session_state['resume_text'] = resume_text
            st.success("✅ Resume uploaded successfully.")
    job_field = st.text_input("💼 Enter Job Field (e.g., Data Science, Software Engineer)", key="job_field")
    if st.button("🔎 Search Recent Jobs"):
        with st.spinner("⏳ Searching for jobs... Please wait"):
            if not resume_text and not job_field:
                st.error("⚠ Please upload a resume or enter a job field.")
            else:
                # Pass state as a dictionary
                state = {
                    "resume_text": resume_text,
                    "job_field": job_field,
                    "skills": [],
                    "job_listings": []
                }
                logging.debug(f"Invoking job_search_graph with initial state: {state}")
                try:
                    result = job_search_graph.invoke(state)
                    logging.debug(f"Job search graph result: {result}")
                    if result.get("job_listings") and "error" not in result.get("job_listings", [{}])[0]:
                        st.markdown("### 🎉 Recent Job Listings")
                        for job in result.get("job_listings", []):
                            st.markdown(f"**{job['title']}**")
                            st.markdown(f"**Company:** {job['company']}")
                            st.markdown(f"**Location:** {job['location']}")
                            st.markdown(f"**Apply:** {job['apply_link']}", unsafe_allow_html=True)
                            st.markdown("---")
                    else:
                        st.error(result.get("job_listings", [{}])[0].get("error", "No jobs found."))
                except Exception as e:
                    logging.error(f"Error in job search graph: {str(e)}")
                    st.error(f"Error in job search: {str(e)}")

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