import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
import io
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize summarizer with caching
@st.cache_resource
def load_summarizer():
    """Load T5 summarizer model with caching"""
    try:
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=512,
            min_length=30,
            do_sample=False
        )
        return summarizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}. Using fallback.")
        return None

# Text extraction functions
def extract_pdf_text(file):
    """Extract text from PDF with error handling"""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"PDF extraction failed: {str(e)}")
        return None

def extract_docx_text(file):
    """Extract text from DOCX"""
    try:
        doc = Document(io.BytesIO(file.read()))
        text = []
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text.strip())
        return " ".join(text)
    except Exception as e:
        st.error(f"DOCX extraction failed: {str(e)}")
        return None

def extract_excel_text(file):
    """Extract text from Excel"""
    try:
        df = pd.read_excel(file, engine='openpyxl')
        # Convert all data to string and join
        text = df.astype(str).to_string()
        return text
    except Exception as e:
        st.error(f"Excel extraction failed: {str(e)}")
        return None

# Smart summarization
def summarize_text(text, summarizer, max_length=150):
    """Generate summary with length control and error handling"""
    if not text or len(text) < 50:
        return "Document too short for meaningful summarization."
    
    try:
        # Truncate text if too long
        if len(text) > 4000:
            text = text[:4000] + "..."
        
        # Generate summary
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Summarization failed: {str(e)}. Here's the first 500 chars: {text[:500]}..."

# Main app
def main():
    st.markdown('<h1 class="main-header">📄 Document Summarizer</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model... This takes ~30 seconds on first run."):
        summarizer = load_summarizer()
    
    if not summarizer:
        st.warning("Using fallback mode. Upload a document to test extraction.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=['pdf', 'docx', 'xlsx', 'xls'],
        help="Supports PDF, DOCX, and Excel files"
    )
    
    if uploaded_file is not None:
        # File info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size/1024:.1f} KB")
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        # Extract text
        with st.spinner("Extracting text..."):
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            if uploaded_file.name.lower().endswith('.pdf'):
                text = extract_pdf_text(uploaded_file)
                doc_type = "PDF"
            elif uploaded_file.name.lower().endswith(('.docx', '.doc')):
                text = extract_docx_text(uploaded_file)
                doc_type = "DOCX"
            else:  # Excel
                text = extract_excel_text(uploaded_file)
                doc_type = "Excel"
        
        if text:
            st.success(f"✅ Successfully extracted {len(text)} characters from {doc_type}")
            
            # Show raw text preview (collapsible)
            with st.expander("👁️ View Raw Extracted Text", expanded=False):
                st.text_area("Extracted Content", text[:2000], height=200)
            
            # Generate summary
            if summarizer:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text, summarizer)
                    
                    st.markdown('### 📋 Document Summary')
                    st.markdown(f'<div class="summary-box"><p>{summary}</p></div>', unsafe_allow_html=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"{Path(uploaded_file.name).stem}_summary.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="📥 Download Raw Text",
                    data=text,
                    file_name=f"{Path(uploaded_file.name).stem}_extracted.txt",
                    mime="text/plain"
                )
        else:
            st.error("Failed to extract text from document. Please try another file.")

if __name__ == "__main__":
    main()
