import streamlit as st
import fitz   # PyMuPDF to read PDFs
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import numpy as np
import json

# Page configuration
st.set_page_config(page_title="Multilingual PDF OCR Studio", layout="wide")

st.title("📄 Multilingual PDF OCR Studio (EN / Chinese)")
st.write("Upload a PDF document to extract text using lightweight, high-performance RapidOCR.")

# Sidebar Controls for Advanced Features
st.sidebar.header("⚙️ Processing Settings")
mode = st.sidebar.radio("Extraction Mode", ["Full Document", "Single Page Preview"])
show_confidence = st.sidebar.checkbox("Show Text Confidence Scores", value=False)

# Initialize lightweight RapidOCR engine (cached)
@st.cache_resource
def load_ocr_engine():
    return RapidOCR()

with st.spinner("Loading OCR engine..."):
    ocr_engine = load_ocr_engine()

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

if uploaded_file is not None:
    # Read the PDF using PyMuPDF
    pdf_bytes = uploaded_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)

    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", total_pages)
    col2.metric("File Name", uploaded_file.name)
    col3.metric("Engine", "RapidOCR (ONNX)")

    st.divider()

    if mode == "Single Page Preview":
        page_num = st.sidebar.slider("Select Page to Preview", 1, total_pages, 1) - 1
        page = pdf_document[page_num]
        
        # Render page preview
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        col_img, col_txt = st.columns(2)
        with col_img:
            st.subheader(f"Page {page_num + 1} Preview")
            st.image(img, use_container_width=True)
            
        with col_txt:
            st.subheader("Extracted Text")
            if st.button("Extract This Page Now"):
                with st.spinner("Running OCR..."):
                    img_np = np.array(img)
                    result, _ = ocr_engine(img_np)
                    if result:
                        page_text = "\n".join([line[1] for line in result])
                        st.text_area("Result", value=page_text, height=300)
                        
                        # Direct text download for single page
                        st.download_button(
                            label="📥 Download Page Text as .txt",
                            data=page_text,
                            file_name=f"page_{page_num + 1}_text.txt",
                            mime="text/plain"
                        )
                        
                        if show_confidence:
                            st.write("**Confidence Scores:**")
                            for line in result:
                                st.caption(f"- `{line[1]}` (Confidence: {line[2]:.2f})")
                    else:
                        st.info("No text detected on this page.")

    else: # Full Document Mode
        if st.button("🚀 Start Full Document OCR Extraction", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            extracted_full_text = ""
            structured_data = []
            
            for page_num in range(total_pages):
                status_text.text(f"Processing page {page_num + 1} of {total_pages}...")
                page = pdf_document[page_num]
                
                # Convert PDF page to an image
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                
                # Run OCR on the page image
                result, _ = ocr_engine(img_np)
                
                page_lines = []
                page_text = ""
                if result:
                    page_text = "\n".join([line[1] for line in result])
                    if show_confidence:
                        page_lines = [{"text": line[1], "confidence": float(line[2])} for line in result]
                
                extracted_full_text += f"\n\n--- Page {page_num + 1} ---\n\n" + page_text
                structured_data.append({"page": page_num + 1, "content": page_text, "lines": page_lines})
                
                # Update progress bar
                progress_bar.progress((page_num + 1) / total_pages)
                
            status_text.text("OCR Processing Complete!")
            st.success("All pages successfully processed!")
            
            # --- PROMINENT EXPORT SECTION ---
            st.markdown("### 📥 Export Options")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.download_button(
                    label="📄 Download Full Text (.txt)",
                    data=extracted_full_text,
                    file_name="extracted_ocr_text.txt",
                    mime="text/plain",
                    type="primary"
                )
            with export_col2:
                json_string = json.dumps(structured_data, ensure_ascii=False, indent=4)
                st.download_button(
                    label="📊 Download Structured Data (.json)",
                    data=json_string,
                    file_name="extracted_ocr_data.json",
                    mime="application/json"
                )
            
            st.divider()

            # Display results in tabs for viewing
            tab1, tab2 = st.tabs(["📝 Plain Text View", "📊 JSON Structure Preview"])
            
            with tab1:
                st.text_area("Full Document Result", value=extracted_full_text, height=350)
                
            with tab2:
                st.code(json_string, language="json")
