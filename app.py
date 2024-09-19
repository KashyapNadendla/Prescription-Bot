import streamlit as st
from google.cloud import vision_v1p3beta1 as vision
import os
import io
import spacy
import fitz  # PyMuPDF for PDF extraction
from openai import OpenAI
import en_core_sci_sm

# Load SciSpaCy model
nlp = en_core_sci_sm.load()

# Set up Google Cloud Vision API Client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"
client = vision.ImageAnnotatorClient()




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


# Function to detect handwritten text using Google Vision API
def detect_handwritten_ocr_image(image_path):
    """Detects handwritten text in a locally uploaded image file."""
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en-t-i0-handwrit"])

    response = client.document_text_detection(image=image, image_context=image_context)

    if response.error.message:
        raise Exception(f"Error: {response.error.message}")

    return response.full_text_annotation.text


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
            st.write("LLM Verification Result:", verification_result)
