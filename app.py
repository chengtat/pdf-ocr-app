import streamlit as st
import fitz   # PyMuPDF to read PDFs
import easyocr
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Multilingual PDF OCR Reader", layout="centered")

st.title("📄 Multilingual Scanned PDF OCR App")
st.write("Upload a scanned PDF document and extract text using OCR supporting English, Simplified Chinese, and Traditional Chinese.")

# Sidebar for language selection
st.sidebar.header("OCR Settings")
selected_langs = st.sidebar.multiselect(
    "Select Languages",
    options=["en", "ch_sim", "ch_tra"],
    default=["en", "ch_sim", "ch_tra"],
    format_func=lambda x: {
        "en": "English",
        "ch_sim": "Chinese (Simplified / 简体中文)",
        "ch_tra": "Chinese (Traditional / 繁體中文)"
    }[x]
)

# Load EasyOCR model (cached with language parameter so it updates if selection changes)
@st.cache_resource
def load_ocr_reader(langs):
    # Ensure langs is passed as a tuple/list for caching
    return easyocr.Reader(list(langs))

if not selected_langs:
    st.warning("Please select at least one language from the sidebar to proceed.")
else:
    with st.spinner("Loading OCR engine... This may take a moment on the first run."):
        reader = load_ocr_reader(tuple(selected_langs))

    # File uploader widget
    uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

    if uploaded_file is not None:
        st.success("PDF uploaded successfully!")
        
        # Read the PDF using PyMuPDF
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        total_pages = len(pdf_document)
        st.info(f"Total pages found in PDF: {total_pages}")
        
        extracted_full_text = ""
        
        # Process button
        if st.button("Start OCR Extraction"):
            progress_bar = st.progress(0)
            
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                
                # Convert PDF page to an image (higher DPI helps with Chinese characters)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Convert image to numpy array for EasyOCR
                img_np = np.array(img)
                
                # Run OCR on the page image
                results = reader.readtext(img_np, detail=0)
                page_text = "\n".join(results)
                
                extracted_full_text += f"\n\n--- Page {page_num + 1} ---\n\n" + page_text
                
                # Update progress bar
                progress_bar.progress((page_num + 1) / total_pages)
                
            st.success("OCR Processing Complete!")
            
            # Display the result in a text area box
            st.subheader("Extracted Text Output:")
            st.text_area("Result", value=extracted_full_text, height=300)
            
            # Download button for the text file
            st.download_button(
                label="Download Extracted Text as .txt",
                data=extracted_full_text,
                file_name="extracted_ocr_text.txt",
                mime="text/plain"
            )
