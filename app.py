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

def get_gemini_response(input_text, pdf_content, prompt):
    """Generate a response using Google Gemini API."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    content = [input_text, prompt]
    if pdf_content:
        content.insert(1, pdf_content[0])
    response = model.generate_content(content)
    return response.text


def input_pdf_setup(uploaded_file):
    """Convert first page of uploaded PDF to an image and encode as base64."""
    if uploaded_file is not None:
        uploaded_file.seek(0)
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        if not images:
            raise ValueError("No pages found in the uploaded PDF")
        first_page = images[0]

        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }]
        return pdf_parts
    else:
        raise FileNotFoundError("No File Uploaded")


def generate_pdf(content):
    """Generate a well-structured PDF document."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=20,
        alignment=1
    )

    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    story = [Paragraph(content, body_style)]
    doc.build(story)
    buffer.seek(0)
    return buffer

st.set_page_config(page_title="A5 ATS Resume Expert")
st.header("MY A5 PERSONAL ATS")

input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])

if uploaded_file:
    st.success("PDF Uploaded Successfully.")

submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")
submit4 = st.button("Personalized Learning Path")
submit5 = st.button("Generate Updated Resume")
submit6 = st.button("Generate 30 Interview Questions")

# Allow user to choose a custom duration for the learning path
learning_duration = st.number_input("Select the duration of the learning path (in months):", min_value=1, max_value=12, value=6)

start_month = st.selectbox("Select starting month for learning path:", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

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

input_prompt4 = f"""
You are an experienced learning coach and technical expert. Create a {learning_duration}-month personalized study plan for an individual aiming to excel in [Job Role],
focusing on the skills, topics, and tools specified in the provided job description. Ensure the study plan includes:
- A list of topics and tools for each month starting from {start_month}.
- Suggested resources (books, online courses, documentation).
- Recommended practical exercises or projects.
- Periodic assessments or milestones.
- Tips for real-world applications.
"""

input_prompt5 = """
You are an experienced resume writer specializing in tech roles. Enhance the provided resume based on the job description to maximize its ATS score.
Ensure proper formatting, use of relevant keywords, and a clean, professional layout with distinct sections and proper spacing.
"""

input_prompt6 = """
Generate a list of 30 targeted interview questions for the specified job role based on the job description provided.
Cover technical skills, soft skills, and scenario-based questions relevant to the role.
"""

if submit1 and uploaded_file:
    pdf_content = input_pdf_setup(uploaded_file)
    response = get_gemini_response(input_prompt1, pdf_content, input_text)
    st.subheader("The Response is:")
    st.write(response)

elif submit3 and uploaded_file:
    pdf_content = input_pdf_setup(uploaded_file)
    response = get_gemini_response(input_prompt3, pdf_content, input_text)
    st.subheader("The Response is:")
    st.write(response)

elif submit4 and uploaded_file:
    pdf_content = input_pdf_setup(uploaded_file)
    response = get_gemini_response(input_prompt4, pdf_content, input_text)
    st.subheader("Personalized Learning Path:")
    st.write(response)

elif submit5 and uploaded_file:
    pdf_content = input_pdf_setup(uploaded_file)
    response = get_gemini_response(input_prompt5, pdf_content, input_text)
    st.subheader("Updated Resume:")
    st.write(response)

    pdf_buffer = generate_pdf(response)
    st.download_button(label="Download Updated Resume", data=pdf_buffer, file_name="Updated_Resume.pdf", mime="application/pdf")

elif submit6:
    response = get_gemini_response(input_prompt6, [], input_text)
    st.subheader("30 Interview Questions:")
    st.write(response)

    pdf_buffer = generate_pdf(response)
    st.download_button(label="Download Interview Questions", data=pdf_buffer, file_name="Interview_Questions.pdf", mime="application/pdf")
