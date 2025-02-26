from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


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
        uploaded_file.seek(0)  # Reset file pointer
        images = pdf2image.convert_from_bytes(uploaded_file.read())  # Removed poppler_path
        first_page = images[0]

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()  # Encode to base64
        }]
        return pdf_parts
    else:
        raise FileNotFoundError("No File Uploaded")

# Streamlit App
st.set_page_config(page_title="A5 ATS Resume Expert")
st.header("MY A5 PERSONAL ATS")

input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])

if uploaded_file:
    st.success("PDF Uploaded Successfully.")

submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")
submit4 = st.button("Personalized Learning Path")
submit5 = st.button("Generate Tailored Resume")

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
You are an expert resume writer with deep knowledge of industry requirements for Data Science, Full Stack, Web Development, Big Data Engineering, DevOps, and Data Analysis.
Using the provided job description and the uploaded resume's content, generate a professional and tailored resume highlighting relevant skills, experience, and achievements.
Ensure the resume is well-structured, ATS-friendly, and optimized to score highly on ATS systems.
"""

generated_resume = None

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
    if uploaded_file and input_text:
        pdf_content = input_pdf_setup(uploaded_file)
        generated_resume = get_gemini_response(input_text, pdf_content, input_prompt5)
        st.subheader("Generated Resume:")
        st.write(generated_resume)

        # Convert generated resume to PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = [Paragraph(line, styles['Normal']) for line in generated_resume.split('\n')]
        doc.build(flowables)

        # Create a download button for the generated resume PDF
        pdf_buffer.seek(0)
        b64 = base64.b64encode(pdf_buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="tailored_resume.pdf">Download Tailored Resume (PDF)</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Please provide a job description and upload a resume.")
