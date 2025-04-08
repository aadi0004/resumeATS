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

# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

def get_gemini_response(prompt):
    """Generate a response using Google Gemini API."""
    if not prompt.strip():
        return "Error: Prompt is empty. Please provide a valid prompt."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, f"Add unique variations each time this prompt is called: {os.urandom(8).hex()}"])
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return f"Error: {str(e)}"



















st.set_page_config(page_title="ResumeSmartX - AI ATS", page_icon="📄", layout='wide')

# Header with a fresh style
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>MY PERSONAL ATS</h1>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# Input section with better layout
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("📋 Job Description:", key="input", height=150)

uploaded_file = None
resume_text = ""
with col2:
    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)...", type=['pdf'])
    if uploaded_file:
        st.success("✅ PDF Uploaded Successfully.")
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                if page and page.extract_text():
                    resume_text += page.extract_text()
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {str(e)}")

# Always visible buttons styled
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠 Quick Actions</h3>", unsafe_allow_html=True)

# Full-width response area
response_container = st.container()




# Ensure response container takes full width
# with st.expander("📋 Response", expanded=True):
#     response_container = st.empty()

# Button actions
if st.button("📖 Tell Me About the Resume"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text:
            response = get_gemini_response(f"Please review the following resume and provide a detailed evaluation: {resume_text}")
            st.write(response)
            st.download_button("💾 Download Resume Evaluation", response, "resume_evaluation.txt")
        else:
            st.warning("⚠ Please upload a valid resume first.")

if st.button("📊 Percentage Match"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text and input_text:
            response = get_gemini_response(f"Evaluate the following resume against this job description and provide a percentage match in first :\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            st.write(response)
            st.download_button("💾 Download Percentage Match", response, "percentage_match.txt")
        else:
            st.warning("⚠ Please upload a resume and provide a job description.")

learning_path_duration = st.selectbox("📆 Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])
if st.button("🎓 Personalized Learning Path"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text and input_text and learning_path_duration:
            response = get_gemini_response(f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text} and also suggest books and other important thing")
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
            response = get_gemini_response(f"Suggest improvements and generate an updated resume for this candidate according to job description, not more than 2 pages:\n{resume_text}")
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
            response = get_gemini_response("Generate 30 technical interview questions and their detailed answers according to that job description.")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")


if st.button("🚀 Skill Development Plan"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text and input_text:
            response = get_gemini_response(f"Based on the resume and job description, suggest courses, books, and projects to improve the candidate's weak or missing skills.\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")

if st.button("🎥 Mock Interview Questions"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text and input_text:
            response = get_gemini_response(f"Generate follow-up interview questions based on the resume and job description, simulating a live interview.\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")

import json

if st.button("💡 AI-Driven Insights"):
    with st.spinner("🔍 Analyzing... Please wait"):
        if resume_text:
            recommendations = get_gemini_response(f"Based on this resume, suggest specific job roles the user is most suited for and analyze market trends for their skills.\n\nResume:\n{resume_text}")
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


# Custom Styling for Buttons
st.markdown("""
    <style>
    div.stButton > button {
        border-radius: 8px;
        padding: 12px 25px;
        margin: 5px;
        font-size: 16px;
        font-weight: bold;
        background-color: #222;
        color: white;
        border: 1px solid #555;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #444;
        border: 1px solid #999;
    }
    </style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("---")
st.markdown("<h2 style='text-align: center; color:#FFA500;'>🚀 MNC Data Science Preparation</h2>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if "selected_mnc" not in st.session_state:
    st.session_state["selected_mnc"] = None

# MNCs List
mnc_data = [
    {"name": "TCS", "color": "#FFA500", "icon": "🎯"},
    {"name": "Infosys", "color": "#03A9F4", "icon": "🚀"},
    {"name": "Wipro", "color": "#9C27B0", "icon": "🔍"},
]

# Layout for MNC Selection in One Row
col1, col2, col3 = st.columns(3)
for col, mnc in zip([col1, col2, col3], mnc_data):
    with col:
        if st.button(f"{mnc['icon']} {mnc['name']}", key=f"{mnc['name']}_button"):
            st.session_state["selected_mnc"] = mnc["name"]

# Display Full-Page Response for Selected MNC
if st.session_state["selected_mnc"]:
    selected_mnc = st.session_state["selected_mnc"]
    
    st.markdown(f"<h3 style='color: #FFA500; text-align: center;'>{selected_mnc} Data Science Preparation</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.spinner("⏳ Analyzing your resume... Please wait"):
        if resume_text:
            response = get_gemini_response(f"Based on the candidate's qualifications and resume, what additional skills and knowledge are needed to secure a Data Science role at {selected_mnc}?")
            st.info(response)
        else:
            st.warning("⚠ Please upload a resume first.")

    # Additional Sections
    if st.button("📂 Project Types & Required Skills"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text:
                response = get_gemini_response(f"What types of Data Science projects does {selected_mnc} typically work on, and what skills align best?")
                st.success(response)
            else:
                st.warning("⚠ Please upload a resume first.")

    if st.button("🛠 Required Skills"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text:
                response = get_gemini_response(f"What key technical and soft skills are needed for a Data Science role at {selected_mnc}?")
                st.success(response)
            else:
                st.warning("⚠ Please upload a resume first.")

    if st.button("💡 Career Recommendations"):
        with st.spinner("⏳ Loading... Please wait"):
            if resume_text:
                response = get_gemini_response(f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at {selected_mnc}?")
                st.success(response)
            else:
                st.warning("⚠ Please upload a resume first.")




st.markdown("---")


st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠 DSA for Data Science</h3>", unsafe_allow_html=True)

 # Main DSA Questions button
level = st.selectbox("📚 Select Difficulty Level:", ["Easy", "Intermediate", "Advanced"])

if st.button(f"📝 Generate {level} DSA Questions (Data Science)"):
    with st.spinner("⏳ Loading... Please wait"):
        response = get_gemini_response(f"Generate 10 DSA questions and answers for data science at {level} level.")
        st.write(response)

topic = st.selectbox("🗂 Select DSA Topic:", ["Arrays", "Linked Lists", "Trees", "Graphs", "Dynamic Programming", "Recursion","algorithm complexity (Big O notation)","sorting" , "searching"])

if st.button(f"📖 Teach me {topic} with Case Studies"):
    with st.spinner("⏳ Gathering resources... Please wait"):
        explanation_response = get_gemini_response(f"Explain the {topic} topic in an easy-to-understand way suitable for beginners, using simple language and clear examples add all details like defination exampales of {topic} and code implementation in python with full explaination of that code.")
        st.write(explanation_response)

        case_study_response = get_gemini_response(f"Provide a real-world case study on {topic} for data science/ data engineer/ m.l/ai with a detailed, easy-to-understand solution.")
        st.write(case_study_response)



st.markdown("---")


question_category = st.selectbox("❓ Select Question Category:", ["Python", "Machine Learning", "Deep Learning", "Docker", "Data Warehousing", "Data Pipelines", "Data Modeling", "SQL"])

if st.button(f"📝 Generate 30 {question_category} Interview Questions"):
    with st.spinner("⏳ Loading... Please wait"):
        response = get_gemini_response(f"Generate 30 {question_category} interview questions and detailed answers")
        st.write(response)
        st.write(response)














import streamlit as st

# Mock AI response function (Replace this with your actual API call)
def get_gemini_response(prompt):
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
        st.session_state.questions = get_gemini_response(selected_topic)
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


st.markdown("---")
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
st.markdown(custom_css, unsafe_allow_html=True)
