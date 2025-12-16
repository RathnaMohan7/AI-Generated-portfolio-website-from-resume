import streamlit as st
import langchain
import PyPDF2
import docx
import zipfile
import io
import re
import dotenv
from dotenv import load_dotenv
load_dotenv()
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage



# ==============================
# Streamlit Config
# ==============================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini")


st.set_page_config(
    page_title="AI Portfolio Generator",
    layout="wide"
)

st.title("🚀 AI-Powered Portfolio Website Generator")


# ==============================
# LLM Configuration
# ==============================

def get_llm(model_name: str = "gemini-2.5-flash-lite") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model_name)


# ==============================
# Resume Text Extraction
# ==============================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    document = docx.Document(file)
    return "\n".join([para.text for para in document.paragraphs])


def extract_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        raise ValueError("Unsupported file format")


# ==============================
# LLM 1 – Resume → Website Spec
# ==============================
RESUME_TO_SPEC_SYSTEM = """
You are an API that outputs ONLY valid JSON.

Rules:
- Output MUST be valid JSON
- Use double quotes only
- No trailing commas
- No explanations
- No markdown
- No extra text

JSON schema:
{
  "name": string,
  "skills": array of strings,
  "experience": array of strings,
  "projects": array of strings,
  "achievements": array of strings,
  "education": array of strings,
  "design_style": string
}
"""

import json

def safe_json_parse(llm_output: str):
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        # Attempt to extract JSON block
        start = llm_output.find("{")
        end = llm_output.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(llm_output[start:end])
            except json.JSONDecodeError:
                return None
        return None

def resume_to_spec(resume_text):
    llm = get_llm()
    messages = [
        SystemMessage(content=RESUME_TO_SPEC_SYSTEM),
        HumanMessage(content=resume_text)
    ]
    response = llm.invoke(messages)
    return response.content


# ==============================
# LLM 2 – Spec → Website Code
# ==============================
SPEC_TO_CODE_SYSTEM = """
You are a senior frontend developer.

Using the portfolio specification, generate a complete portfolio website.

Rules:
- Output ONLY valid code
- Separate files using EXACT markers:
  <!-- index.html -->
  <!-- style.css -->
  <!-- script.js -->
- Use responsive modern design
- No markdown
"""

def spec_to_website_code(spec_json):
    llm = get_llm()
    messages = [
        SystemMessage(content=SPEC_TO_CODE_SYSTEM),
        HumanMessage(content=spec_json)
    ]
    response = llm.invoke(messages)
    return response.content


# ==============================
# ZIP Packaging
# ==============================
def create_zip(files: dict):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        for filename, content in files.items():
            zipf.writestr(filename, content)
    buffer.seek(0)
    return buffer


# ==============================
# Streamlit UI
# ==============================
uploaded_file = st.file_uploader(
    "📄 Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

if uploaded_file:
    with st.spinner("🔍 Extracting resume text..."):
        resume_text = extract_resume_text(uploaded_file)

    st.success("Resume text extracted successfully")

    with st.expander("📃 View Resume Text"):
        st.text(resume_text)

    if st.button("✨ Generate Portfolio Website"):
        # -------- LLM 1 --------
        with st.spinner("🤖 LLM 1: Analyzing resume..."):
            raw_spec = resume_to_spec(resume_text)
            spec_data = safe_json_parse(raw_spec)

            if spec_data is None:
                st.error("❌ LLM returned invalid JSON. Please try again.")
                st.stop()

            st.subheader("🧠 Extracted Portfolio Specification")
            st.json(spec_data)

        # -------- LLM 2 --------
        with st.spinner("🎨 LLM 2: Generating website code..."):
            website_code = spec_to_website_code(json.dumps(spec_data))

        # -------- Split Files --------
        index_html = re.search(
            r"<!-- index.html -->(.*?)<!-- style.css -->",
            website_code,
            re.S
        ).group(1)

        style_css = re.search(
            r"<!-- style.css -->(.*?)<!-- script.js -->",
            website_code,
            re.S
        ).group(1)

        script_js = re.search(
            r"<!-- script.js -->(.*)",
            website_code,
            re.S
        ).group(1)

        # -------- Preview --------
        st.subheader("🌐 Live Website Preview")
        st.components.v1.html(index_html, height=900, scrolling=True)

        # -------- Download ZIP --------
        zip_buffer = create_zip({
            "index.html": index_html,
            "style.css": style_css,
            "script.js": script_js
        })

        st.download_button(
            "⬇️ Download Portfolio Website (ZIP)",
            data=zip_buffer,
            file_name="portfolio_website.zip",
            mime="application/zip"
        )
