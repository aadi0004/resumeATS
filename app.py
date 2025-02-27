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
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    """Convert first page of uploaded PDF to an image and encode as base64."""
    if uploaded_file is not None:
        uploaded_file.seek(0)
        images = pdf2image.convert_from_bytes(uploaded_file.read())
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

def generate_pdf(resume_content):
    """Generate a well-structured and ATS-optimized PDF resume with proper headings, subheadings, and content."""
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

    section_heading_style = ParagraphStyle(
        'SectionHeadingStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12
    )

    subheading_style = ParagraphStyle(
        'SubheadingStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    story = []

    for section in resume_content.split('\n'):
        if section.strip():
            if section.startswith('**'):
                story.append(Paragraph(f'<b>{section.strip().replace("**", "")}</b>', section_heading_style))
            elif ':' in section:
                subheading, text = section.split(':', 1)
                story.append(Paragraph(f'<b>{subheading.strip()}</b>', subheading_style))
                story.append(Paragraph(text.strip(), body_style))
            else:
                story.append(Paragraph(section.strip(), body_style))
            story.append(Spacer(1, 8))

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


input_prompt4 = """
You are an experienced learning coach and technical expert. Create a 6-month personalized study plan for an individual aiming to excel in [Job Role],
focusing on the skills, topics, and tools specified in the provided job description. Ensure the study plan includes:
- A list of topics and tools for each month.
- Suggested resources (books, online courses, documentation).
- Recommended practical exercises or projects.
- Periodic assessments or milestones.
- Tips for real-world applications.
"""

input_prompt5 = """
You are an experienced resume writer specializing in tech roles. Enhance the provided resume based on the job description to maximize its ATS score.
Ensure proper formatting, use of relevant keywords, and a clean, professional layout with distinct sections and proper spacing.
"""

if submit1:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

elif submit3:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

elif submit4:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt4, pdf_content, input_text)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")


elif submit5:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt5, pdf_content, input_text)
        st.subheader("Updated Resume:")
        st.write(response)

        pdf_buffer = generate_pdf(response)
        st.download_button(label="Download Updated Resume", data=pdf_buffer, file_name="Updated_Resume.pdf", mime="application/pdf")
    else:
        st.warning("Please upload a resume.")
