from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai
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

input_prompt1 = """
You are an experienced HR with tech expertise in Data Science, Full Stack, Web Development, Big Data Engineering, DevOps, or Data Analysis.
Your task is to review the provided resume against the job description for these roles.
Please evaluate the candidate's profile, highlighting strengths and weaknesses in relation to the specified job role.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with expertise in Data Science, Full Stack, Web Development, Big Data Engineering, DevOps, and Data Analysis.
Your task is to evaluate the resume against the job description. Provide:
1. The percentage match.
2. Keywords missing.
3. Final evaluation.
"""

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

st.set_page_config(page_title="A5 ATS Resume Expert")
st.header("MY A5 PERSONAL ATS")

input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])

if uploaded_file:
    st.success("PDF Uploaded Successfully.")

# Always visible buttons
st.button("Tell Me About the Resume")
st.button("Percentage Match")
st.button("Personalized Learning Path")
st.button("Generate Updated Resume")
st.button("Generate 30 Interview Questions and Answers")

# New dropdown for personalized learning path duration
learning_path_duration = st.selectbox("Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])

# Dropdown for selecting interview question category
question_category = st.selectbox("Select Question Category:", ["Python", "Machine Learning", "Deep Learning", "Docker", "Data Warehousing", "Data Pipelines", "Data Modeling", "SQL"])

# Show only one button based on selected category
if question_category == "Python":
    if st.button("30 Python Interview Questions"):
        response = get_gemini_response("Generate 30 Python interview questions and detailed answers")
        if not response.startswith("Error"):
            st.subheader("Python Interview Questions and Answers:")
            st.write(response)
            st.download_button("Download Python Questions", response, f"python_questions_{os.urandom(4).hex()}.txt")
        else:
            st.error(response)

elif question_category == "Machine Learning":
    if st.button("30 Machine Learning Interview Questions"):
        response = get_gemini_response("Generate 30 Machine Learning interview questions and detailed answers")
        if not response.startswith("Error"):
            st.subheader("Machine Learning Interview Questions and Answers:")
            st.write(response)
            st.download_button("Download ML Questions", response, f"ml_questions_{os.urandom(4).hex()}.txt")
        else:
            st.error(response)

elif question_category == "Deep Learning":
    if st.button("30 Deep Learning Interview Questions"):
        response = get_gemini_response("Generate 30 Deep Learning interview questions and detailed answers")
        if not response.startswith("Error"):
            st.subheader("Deep Learning Interview Questions and Answers:")
            st.write(response)
            st.download_button("Download DL Questions", response, f"dl_questions_{os.urandom(4).hex()}.txt")
        else:
            st.error(response)

elif question_category == "Docker":
    if st.button("30 Docker Interview Questions"):
        response = get_gemini_response("Generate 30 Docker interview questions and detailed answers")
        if not response.startswith("Error"):
            st.subheader("Docker Interview Questions and Answers:")
            st.write(response)
            st.download_button("Download Docker Questions", response, f"docker_questions_{os.urandom(4).hex()}.txt")
        else:
            st.error(response)

# Data Engineering questions
elif question_category in ["Data Warehousing", "Data Pipelines", "Data Modeling", "SQL"]:
    if st.button(f"30 {question_category} Interview Questions"):
        response = get_gemini_response(f"Generate 30 {question_category} interview questions and detailed answers")
        if not response.startswith("Error"):
            st.subheader(f"{question_category} Interview Questions and Answers:")
            st.write(response)
            st.download_button(f"Download {question_category} Questions", response, f"{question_category.lower().replace(' ', '_')}_questions_{os.urandom(4).hex()}.txt")
        else:
            st.error(response)
