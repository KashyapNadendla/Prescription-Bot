import os
import json
import base64
import tempfile

import streamlit as st
import fitz  # PyMuPDF
import en_core_sci_sm
from openai import OpenAI
from google.oauth2 import service_account
from google.cloud import vision_v1p3beta1 as vision
from dotenv import load_dotenv

# ─── 1. Configuration & Clients ─────────────────────────────────────────────────

load_dotenv()

REQUIRED_ENVS = ["OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS_JSON"]
for var in REQUIRED_ENVS:
    if not os.getenv(var):
        st.error(f"Missing env var: {var}")
        st.stop()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
nlp = en_core_sci_sm.load()

gcp_json = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(gcp_json)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# ─── 2. Utility Functions ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def detect_handwritten_text(path_or_uri: str) -> str:
    """Extract handwritten text via OpenAI Vision (local) or Google Vision (GCS URI)."""
    try:
        if path_or_uri.startswith("gs://"):
            resp = vision_client.document_text_detection({"image_uri": path_or_uri})
            return resp.full_text_annotation.text or ""
        # local file → OpenAI Vision
        with open(path_or_uri, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"Extract only the handwritten prescription text."},
                {"role":"user","content":[
                    {"type":"text","text":"Please extract the handwritten prescription."},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ],
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def call_openai_system(user_content: str, system_prompt: str) -> str:
    """Generic wrapper for OpenAI chat completion with error handling."""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_content}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return ""

def generate_summaries(history: dict, pdf_text: str) -> tuple[str, str]:
    combined = json.dumps(history, indent=2) + "\n\n" + pdf_text
    pat = call_openai_system(
        combined,
        "You are a caring assistant. Summarize this patient history in plain, reassuring language (2–3 paragraphs)."
    )
    doc = call_openai_system(
        combined,
        "You are a medical scribe. Produce a concise, structured clinical note in SOAP format."
    )
    return pat, doc

def verify_prescription(pres_text: str, full_history: str) -> str:
    prompt = f"Patient History:\n{full_history}\n\nPrescription:\n{pres_text}\n\n" + \
             "Analyze for dosing errors, interactions, missing instructions; output a structured risk & recommendations report."
    return call_openai_system(prompt, "You are an expert in prescription validation. Provide risk analysis & recommendations.")

# ─── 3. Session State Helpers ────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = {}
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "pres_text" not in st.session_state:
    st.session_state.pres_text = ""

def reset_all():
    for key in ["history", "pdf_text", "pres_text"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()

# ─── 4. Layout & Navigation ─────────────────────────────────────────────────────

st.set_page_config(page_title="Medical Intake & RX Checker", layout="wide")
with st.sidebar:
    st.title("Navigation")
    phase = st.radio("", ["1️⃣ Intake", "2️⃣ Summaries", "3️⃣ Verification"])
    st.markdown("---")
    if st.button("🔄 Start Over"):
        reset_all()

# ─── 5. Phase 1: Intake ──────────────────────────────────────────────────────────

if phase.startswith("1"):
    st.header("🩺 Medical History Intake")

    with st.form("history_form", clear_on_submit=False):
        st.text_input("Full name", key="full_name", value=st.session_state.history.get("full_name", ""))
        st.number_input("Age", min_value=0, max_value=120, key="age", value=st.session_state.history.get("age", 0))
        st.selectbox("Gender", ["Prefer not to say","Female","Male","Other"], key="gender", index=0)
        st.text_area("Allergies", key="allergies", value=st.session_state.history.get("allergies",""))
        st.text_area("Current medications", key="medications", value=st.session_state.history.get("medications",""))
        st.text_area("Past conditions or surgeries", key="conditions", value=st.session_state.history.get("conditions",""))
        st.text_area("Current symptoms", key="symptoms", value=st.session_state.history.get("symptoms",""))
        submitted = st.form_submit_button("Save History")
        if submitted:
            # Copy form fields into session_state.history
            for field in ["full_name","age","gender","allergies","medications","conditions","symptoms"]:
                st.session_state.history[field] = st.session_state[field]
            st.success("✅ History saved! You can now go to Summaries or Verification.")

    if st.session_state.history:
        st.expander("📋 View current history", expanded=False).json(st.session_state.history)

# ─── 6. Phase 2: Summaries ───────────────────────────────────────────────────────

elif phase.startswith("2"):
    if not st.session_state.history:
        st.warning("Please complete the Intake first.")
        st.stop()

    st.header("📝 Generate Summaries")
    st.subheader("Optional: Upload detailed history PDF")
    uploaded_pdf = st.file_uploader("", type="pdf")
    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.read()
        st.session_state.pdf_text = extract_pdf_text(pdf_bytes)
        with st.expander("🔍 Preview extracted PDF text", expanded=False):
            st.text_area("", st.session_state.pdf_text, height=200)

    if st.button("Generate Patient & Doctor Summaries"):
        with st.spinner("Calling LLM…"):
            patient_sum, doctor_sum = generate_summaries(st.session_state.history, st.session_state.pdf_text)
        st.subheader("Patient-Friendly Summary")
        st.write(patient_sum)
        st.subheader("Doctor-Ready Report")
        st.code(doctor_sum, language="text")

# ─── 7. Phase 3: Prescription Verification ───────────────────────────────────────

else:
    if not st.session_state.history:
        st.warning("Please complete the Intake first.")
        st.stop()

    st.header("✔️ Prescription OCR & Verification")
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Input source", ["Upload File", "GCS URI"])
        if mode == "Upload File":
            up = st.file_uploader("", type=["png","jpg","jpeg","pdf"])
            if up:
                data = up.read()
                if up.type == "application/pdf":
                    st.session_state.pres_text = extract_pdf_text(data)
                else:
                    # write to temp file for vision
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
                    tmp.write(data)
                    tmp.close()
                    st.session_state.pres_text = detect_handwritten_text(tmp.name)
        else:
            uri = st.text_input("gs://…", key="gcs_uri")
            if uri:
                st.session_state.pres_text = detect_handwritten_text(uri)

    with col2:
        if st.session_state.pres_text:
            st.subheader("Extracted Prescription")
            st.text_area("", st.session_state.pres_text, height=200)

    if st.session_state.pres_text and st.button("Verify Prescription"):
        full_hist = json.dumps(st.session_state.history, indent=2) + "\n\n" + st.session_state.pdf_text
        with st.spinner("Verifying…"):
            report = verify_prescription(st.session_state.pres_text, full_hist)
        st.subheader("Verification Report")
        st.write(report)