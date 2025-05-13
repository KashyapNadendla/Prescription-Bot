import streamlit as st
from google.cloud import vision_v1p3beta1 as vision
import os
import io
import spacy
import fitz 
from openai import OpenAI
import en_core_sci_sm
import google.generativeai as genai
from dotenv import load_dotenv
import json
from google.oauth2 import service_account
import subprocess
import sys
import base64
from PIL import Image
import requests

load_dotenv()

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# GEMINI_API_KEY = "AIzaSyAdG2ZFnLDq-KxUbJFlut5502rn759UPMM"

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# Load SciSpaCy model
nlp = en_core_sci_sm.load()

# Set up Google Cloud Vision API Client
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"
# client = vision.ImageAnnotatorClient()

google_credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if google_credentials_json:
    credentials_info = json.loads(google_credentials_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_info)

    # Initialize the Vision API client with these credentials
    client = vision.ImageAnnotatorClient(credentials=credentials)
else:
    raise ValueError("Google credentials JSON not found in environment variables.")


# Open AI initialization
client_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Function to verify prescription using LLM (GPT-4)
def verify_prescription_with_llm(extracted_text, patient_history):
    prompt = f"""
    Verify the following prescription for accuracy based on the patient's medical history:
    
    Patient History:
    {patient_history}
    
    Prescription:
    {extracted_text}
    
    Analyze for potential issues like incorrect dosages, drug interactions, and missing instructions, and provide recommendations accordingly.
    """

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an advanced assistant specializing in prescription validation. Use the patient's medical history to analyze the prescription for potential issues like incorrect dosages, drug interactions, and missing instructions. Provide a structured report that includes risk analysis and recommendations."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content



## Gemini
def gemini_prescription_respoonse(extracted_text,patient_history):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Verify the following prescription for accuracy based on the patient's medical history:
    
    Patient History:
    {patient_history}
    
    Prescription:
    {extracted_text}
    
    Analyze for potential issues like incorrect dosages, drug interactions, and missing instructions, and provide recommendations accordingly.
    """
    
    response = model.generate_content(prompt)
    print(response.text)


# Function to detect handwritten text using Google Vision API
# def detect_handwritten_ocr_image(image_path):
#     """Detects handwritten text in a locally uploaded image file."""
#     with io.open(image_path, 'rb') as image_file:
#         content = image_file.read()

#     image = vision.Image(content=content)
#     image_context = vision.ImageContext(language_hints=["en-t-i0-handwrit"])

#     response = client.document_text_detection(image=image, image_context=image_context)

#     if response.error.message:
#         raise Exception(f"Error: {response.error.message}")

#     return response.full_text_annotation.text

def detect_handwritten_ocr_image(image_path):
    """Uses OpenAI's Vision model to extract handwritten text from an image."""

    # Open image and convert to base64
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Send to OpenAI Vision model
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the handwritten prescription from this image and return just the readable text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content


# Function to extract text from PDF (patient history)
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()

    return text


# Streamlit app logic
st.title("Handwritten Prescription OCR and Verification with Patient History")

option = st.selectbox("Choose Input Source:", ["Upload Local File", "Google Cloud Storage URI"])

if option == "Upload Local File":
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, PDF)", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")

        # Extract text from the uploaded image
        extracted_text = detect_handwritten_ocr_image(f"data/{uploaded_file.name}")
        
        # Display extracted text in a larger box
        st.text_area("Extracted Text", extracted_text, height=300)

        # Upload patient history (PDF)
        uploaded_pdf = st.file_uploader("Upload Patient History (PDF)", type=["pdf"])

        patient_history = ""
        if uploaded_pdf is not None:
            # Save the uploaded PDF
            with open(f"data/{uploaded_pdf.name}", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success(f"Patient history file {uploaded_pdf.name} uploaded successfully!")

            # Extract text from the PDF file
            patient_history = extract_text_from_pdf(f"data/{uploaded_pdf.name}")
            st.text_area("Extracted Patient History", patient_history, height=300)

        # Verify prescription using LLM
        if st.button("Verify Prescription with Patient History"):
            verification_result = verify_prescription_with_llm(extracted_text, patient_history)
            # verification_result = gemini_prescription_respoonse(extracted_text, patient_history)
            st.write("LLM Verification Result:", verification_result)

elif option == "Google Cloud Storage URI":
    uri = st.text_input("Enter Google Cloud Storage URI (gs://...):")

    if uri:
        extracted_text = detect_handwritten_ocr_image(uri)
        
        # Display extracted text in a larger box
        st.text_area("Extracted Text", extracted_text, height=300)

        # Upload patient history (PDF)
        uploaded_pdf = st.file_uploader("Upload Patient History (PDF)", type=["pdf"])

        patient_history = ""
        if uploaded_pdf is not None:
            # Save the uploaded PDF
            with open(f"data/{uploaded_pdf.name}", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success(f"Patient history file {uploaded_pdf.name} uploaded successfully!")

            # Extract text from the PDF file
            patient_history = extract_text_from_pdf(f"data/{uploaded_pdf.name}")
            st.text_area("Extracted Patient History", patient_history, height=300)

        # Verify prescription using LLM
        if st.button("Verify Prescription with Patient History"):
            verification_result = verify_prescription_with_llm(extracted_text, patient_history)
            # verification_result = gemini_prescription_respoonse(extracted_text, patient_history)
            st.write("LLM Verification Result:", verification_result)
