import streamlit as st
import fitz  # PyMuPDF to read PDFs
import easyocr
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="PDF OCR Reader (Chinese & English)", layout="centered")

st.title("📄 Scanned PDF OCR App (Chinese / English)")
st.write("Upload a scanned PDF document containing Chinese or English, and this app will automatically extract the text.")

# Load EasyOCR model with Simplified Chinese, Traditional Chinese, and English support
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ch_sim', 'ch_tra', 'en'])

with st.spinner("Loading OCR engine... (This takes a minute on the first run as it downloads language weights)"):
    reader = load_ocr_reader()

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

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
            
            # Convert PDF page to a high-resolution image (dpi=300 for sharp Chinese character strokes)
            pix = page.get_pixmap(dpi=300)
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
