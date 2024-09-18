# Handwritten Prescription OCR using Google Cloud Vision API and Report Validation using OpenAI GPT-4o-mini

This Streamlit application allows users to upload handwritten prescriptions and automatically extract the text using the Google Cloud Vision API for OCR. The extracted text is then validated using OpenAI's GPT-4 model, ensuring accuracy in dosage, drug interactions, and adherence to medical guidelines. The system generates a structured report analyzing potential issues in the prescription.

### Features

- Handwritten Prescription OCR: Extracts text from handwritten images (JPG, PNG, PDF) using Google Cloud Vision's OCR capabilities.
- Prescription Verification: Validates the extracted prescription text using OpenAI’s GPT-4 model. The model checks for potential errors, drug interactions, missing instructions, and more.

### Customizable Input Sources:

- Upload Local Files: Upload prescription images from your local system.
- Google Cloud Storage URI: Provide a Google Cloud Storage URI for processing images stored on the cloud.
- Structured Report Generation: Generates a detailed report with identified errors, risk analysis, and recommendations for improvements.

### Technologies Used

- Streamlit: For building the user interface and handling file uploads.
- Google Cloud Vision API: For detecting and extracting handwritten text from prescription images.
- OpenAI GPT-4o-mini For verifying prescriptions and providing analysis on drug interactions, dosage, and other risks.
- SciSpaCy: For potential drug interaction checks (can be expanded to incorporate patient history and drug databases).

### How It Works

Handwritten OCR:

- The user uploads a prescription image or provides a Google Cloud Storage URI.
- The app uses Google Cloud Vision API to extract the handwritten text.
- Extracted text is displayed in the app for user verification.

Prescription Verification:

- The extracted text is sent to OpenAI GPT-4 for analysis and validation.
- The model checks for prescription errors, missing information, potential drug interactions, and other risks.
- A detailed report is generated with recommendations.
