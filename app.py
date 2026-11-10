import io
import json
import gc
import re
import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from PIL import Image
from gtts import gTTS
from googletrans import Translator
from rapidocr_onnxruntime import RapidOCR

# --- Page Configuration ---
st.set_page_config(
    page_title="Multilingual PDF OCR Studio Pro",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Multilingual PDF OCR Studio Pro (EN / Chinese) + TTS")
st.write("Extract text, analyze tabular data, inspect bounding boxes, translate, and synthesize audio with AI.")

# --- Session State Initialization ---
default_states = {
    "extracted_full_text": "",
    "structured_data": [],
    "total_chinese_chars": 0,
    "total_english_words": 0,
    "single_page_results": {},
    "page_tables": {}
}

for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Cached Engine Loading ---
@st.cache_resource
def load_rapidocr_engine():
    return RapidOCR()

@st.cache_resource
def load_easyocr_engine():
    import easyocr
    return easyocr.Reader(['ch_sim', 'en'])

@st.cache_resource
def get_translator():
    return Translator()

# --- Helper Functions ---
def generate_audio(text: str, lang: str) -> io.BytesIO:
    """Converts text to MP3 audio stream with automatic payload chunking."""
    if not text.strip():
        raise ValueError("Text content is empty.")
    
    max_chunk = 4000
    if len(text) > max_chunk:
        chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
        combined_audio = io.BytesIO()
        for idx, chunk in enumerate(chunks):
            tts = gTTS(text=chunk, lang=lang)
            chunk_fp = io.BytesIO()
            tts.write_to_fp(chunk_fp)
            chunk_fp.seek(0)
            combined_audio.write(chunk_fp.read())
        combined_audio.seek(0)
        return combined_audio
    else:
        audio_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp

def process_ocr(img_np, engine_type="RapidOCR"):
    """Executes OCR scan based on selected engine, returning unified result format."""
    if engine_type == "RapidOCR":
        ocr_engine = load_rapidocr_engine()
        result, _ = ocr_engine(img_np)
        return result
    else: # EasyOCR
        reader = load_easyocr_engine()
        raw_result = reader.readtext(img_np)
        # Format EasyOCR [bbox, text, conf] to match RapidOCR output structure [[bbox], text, conf]
        formatted_result = []
        for box, text, conf in raw_result:
            formatted_box = [[int(pt[0]), int(pt[1])] for pt in box]
            formatted_result.append([formatted_box, text, conf])
        return formatted_result

def process_page(page, dpi=150, engine_type="RapidOCR"):
    """Renders PDF page to array and executes OCR processing."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)
    result = process_ocr(img_np, engine_type)
    pix = None
    return img_np, result

def extract_tables_from_pdf(pdf_bytes, page_idx):
    """Extracts tables from a target page using pdfplumber."""
    tables_data = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if page_idx < len(pdf.pages):
            page = pdf.pages[page_idx]
            extracted_tables = page.extract_tables()
            for t in extracted_tables:
                if t:
                    df = pd.DataFrame(t[1:], columns=t[0])
                    tables_data.append(df)
    return tables_data

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Processing Settings")
engine_choice = st.sidebar.selectbox("OCR Engine Backend", ["RapidOCR", "EasyOCR"])
mode = st.sidebar.radio("Extraction Mode", ["Full Document", "Single Page Preview"])
show_bounding_boxes = st.sidebar.checkbox("Draw Bounding Boxes on Image", value=True)
show_confidence = st.sidebar.checkbox("Show Text Confidence Scores", value=True)

st.sidebar.divider()
st.sidebar.header("🔍 Keyword Search & Highlight")
search_query = st.sidebar.text_input("Search document for text:", placeholder="Type word or pattern...")

# --- Main File Upload Area ---
uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", total_pages)
    col2.metric("File Name", uploaded_file.name)
    col3.metric("Selected Engine", engine_choice)

    st.divider()

    # =========================================================================
    # SINGLE PAGE PREVIEW MODE
    # =========================================================================
    if mode == "Single Page Preview":
        page_num = st.sidebar.slider("Select Page to Preview", 1, total_pages, 1) - 1
        page = pdf_document[page_num]
        
        col_img, col_txt = st.columns(2)
        
        with col_img:
            st.subheader(f"Page {page_num + 1} Visual Output")
            run_preview = st.button("🔍 Run Page OCR", type="primary", key=f"run_page_{page_num}")
            
            if run_preview or page_num in st.session_state.single_page_results:
                if run_preview or page_num not in st.session_state.single_page_results:
                    with st.spinner(f"Running {engine_choice}..."):
                        img_np, result = process_page(page, dpi=150, engine_type=engine_choice)
                        st.session_state.single_page_results[page_num] = {
                            "img_np": img_np,
                            "result": result
                        }
                        # Extract tabular data
                        st.session_state.page_tables[page_num] = extract_tables_from_pdf(pdf_bytes, page_num)
                
                cached_data = st.session_state.single_page_results[page_num]
                img_np = cached_data["img_np"]
                result = cached_data["result"]
                
                if result and show_bounding_boxes:
                    annotated_img = img_np.copy()
                    for idx, line in enumerate(result):
                        box = np.array(line[0], dtype=np.int32)
                        cv2.polylines(annotated_img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                        # Add index overlay number for bounding box inspector
                        cv2.putText(annotated_img, str(idx), (int(box[0][0]), int(box[0][1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    st.image(annotated_img, use_container_width=True)
                else:
                    st.image(img_np, use_container_width=True)
            else:
                pix = page.get_pixmap(dpi=100)
                preview_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                st.image(preview_img, use_container_width=True)

        with col_txt:
            st.subheader("Extracted Text & Speech")
            if page_num in st.session_state.single_page_results:
                result = st.session_state.single_page_results[page_num]["result"]
                
                if result:
                    page_text = "\n".join([line[1] for line in result])
                    
                    # Highlight keyword search matches if entered
                    if search_query.strip():
                        matches = len(re.findall(re.escape(search_query), page_text, re.IGNORECASE))
                        if matches > 0:
                            st.success(f"Found **{matches}** match(es) for '{search_query}' on this page.")
                        else:
                            st.warning(f"No matches for '{search_query}' on this page.")

                    st.text_area("OCR Result Text", value=page_text, height=200, key=f"txt_area_{page_num}")
                    
                    # --- ENHANCEMENT: Interactive Bounding Box Inspector & Inline Correction ---
                    with st.expander("🛠️ Bounding Box Inspector & Text Corrector", expanded=False):
                        line_options = [f"Line {idx}: {line[1]} (Conf: {float(line[2]):.2f})" for idx, line in enumerate(result)]
                        selected_idx = st.selectbox("Select Line to Inspect / Edit:", range(len(result)), format_func=lambda x: line_options[x])
                        
                        target_line = result[selected_idx]
                        st.write(f"**Bounding Box Polygon:** `{target_line[0]}`")
                        
                        corrected_val = st.text_input("Correct Extracted Line Text:", value=target_line[1], key=f"edit_line_{page_num}_{selected_idx}")
                        if st.button("Save Line Correction", key=f"save_line_{page_num}_{selected_idx}"):
                            st.session_state.single_page_results[page_num]["result"][selected_idx][1] = corrected_val
                            st.success("Line corrected! Refreshing...")
                            st.rerun()

                    if show_confidence:
                        with st.expander("🔍 Detailed Confidence Metrics", expanded=False):
                            for line in result:
                                st.write(f"- `{line[1]}` (Confidence: **{float(line[2]):.4f}**)")

                    # --- ENHANCEMENT: Table Detection & CSV Export ---
                    tables = st.session_state.page_tables.get(page_num, [])
                    if tables:
                        st.markdown("### 📊 Extracted Tables on Page")
                        for idx, df in enumerate(tables):
                            st.write(f"**Table {idx + 1}**")
                            st.dataframe(df, use_container_width=True)
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"📥 Download Table {idx + 1} as CSV",
                                data=csv_data,
                                file_name=f"page_{page_num + 1}_table_{idx + 1}.csv",
                                mime="text/csv",
                                key=f"dl_table_{page_num}_{idx}"
                            )

                    # --- ENHANCEMENT: AI Translation Panel ---
                    st.markdown("---")
                    st.markdown("### 🌐 AI Machine Translation")
                    target_lang = st.selectbox(
                        "Target Language",
                        ["zh-cn", "zh-tw", "en", "ja", "fr", "de", "es"],
                        key=f"trans_lang_{page_num}"
                    )
                    if st.button("Translate Text", key=f"btn_trans_{page_num}"):
                        with st.spinner("Translating..."):
                            try:
                                translator = get_translator()
                                translated = translator.translate(page_text, dest=target_lang)
                                st.info(f"**Translation ({target_lang}):**\n\n{translated.text}")
                            except Exception as e:
                                st.error(f"Translation Error: {e}")

                    st.markdown("---")
                    st.markdown("### 🔊 Text-to-Speech")
                    tts_lang = st.selectbox("Select Language", ["zh-CN", "en"], index=0, key=f"tts_lang_{page_num}")
                    
                    if st.button("▶️ Generate Audio", key=f"gen_aud_{page_num}"):
                        with st.spinner("Synthesizing audio..."):
                            try:
                                audio_stream = generate_audio(page_text, tts_lang)
                                audio_bytes = audio_stream.getvalue()
                                
                                st.audio(audio_bytes, format="audio/mp3")
                                st.download_button(
                                    label="📥 Download Audio (.mp3)",
                                    data=audio_bytes,
                                    file_name=f"page_{page_num + 1}_audio.mp3",
                                    mime="audio/mp3",
                                    key=f"dl_aud_btn_{page_num}"
                                )
                            except Exception as e:
                                st.error(f"TTS Error: {e}")

                    st.download_button(
                        label="📥 Download Page Text (.txt)",
                        data=page_text,
                        file_name=f"page_{page_num + 1}_text.txt",
                        mime="text/plain",
                        key=f"dl_txt_btn_{page_num}"
                    )
                else:
                    st.info("No text detected on this page.")
            else:
                st.info("Click 'Run Page OCR' on the left panel to begin scanning.")

    # =========================================================================
    # FULL DOCUMENT MODE
    # =========================================================================
    else:
        if st.button("🚀 Start Full Document OCR Extraction", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            temp_full_text = ""
            temp_structured_data = []
            temp_chinese_chars = 0
            temp_english_words = 0
            
            for page_idx in range(total_pages):
                status_text.text(f"Processing page {page_idx + 1} of {total_pages} using {engine_choice}...")
                page = pdf_document[page_idx]
                
                img_np, result = process_page(page, dpi=150, engine_type=engine_choice)
                
                page_lines = []
                page_text = ""
                
                if result:
                    page_text = "\n".join([line[1] for line in result])
                    for line in result:
                        text_val = line[1]
                        temp_chinese_chars += sum(1 for c in text_val if '\u4e00' <= c <= '\u9fff')
                        temp_english_words += len([w for w in text_val.split() if w.isascii() and w.isalpha()])
                        
                        if show_confidence:
                            page_lines.append({
                                "text": text_val,
                                "confidence": float(line[2])
                            })
                
                temp_full_text += f"\n\n--- Page {page_idx + 1} ---\n\n" + page_text
                temp_structured_data.append({
                    "page": page_idx + 1,
                    "content": page_text,
                    "lines": page_lines
                })
                
                del img_np, result
                gc.collect()
                progress_bar.progress((page_idx + 1) / total_pages)
            
            st.session_state.extracted_full_text = temp_full_text
            st.session_state.structured_data = temp_structured_data
            st.session_state.total_chinese_chars = temp_chinese_chars
            st.session_state.total_english_words = temp_english_words
            
            status_text.text("OCR Processing Complete!")
            st.success("All pages successfully processed!")

        if st.session_state.extracted_full_text:
            # --- Keyword Search Summary Banner ---
            if search_query.strip():
                full_text = st.session_state.extracted_full_text
                total_matches = len(re.findall(re.escape(search_query), full_text, re.IGNORECASE))
                st.info(f"🔍 Found **{total_matches}** total occurrence(s) of **'{search_query}'** across the document.")

            with st.expander("📊 Document Text Analytics", expanded=True):
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                stat_col1.metric("Total Characters", len(st.session_state.extracted_full_text))
                stat_col2.metric("Estimated Chinese Characters", st.session_state.total_chinese_chars)
                stat_col3.metric("Estimated English Words", st.session_state.total_english_words)

            st.markdown("### 📥 Export Options & Text-to-Speech")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.download_button(
                    label="📄 Download Full Text (.txt)",
                    data=st.session_state.extracted_full_text,
                    file_name="extracted_ocr_text.txt",
                    mime="text/plain",
                    type="primary"
                )
            with export_col2:
                json_string = json.dumps(st.session_state.structured_data, ensure_ascii=False, indent=4)
                st.download_button(
                    label="📊 Download Structured Data (.json)",
                    data=json_string,
                    file_name="extracted_ocr_data.json",
                    mime="application/json"
                )
            
            # --- Document-level Translation ---
            with st.expander("🌐 Full Document Translation", expanded=False):
                doc_trans_lang = st.selectbox("Target Translation Language", ["zh-cn", "zh-tw", "en", "ja", "fr", "de", "es"], key="doc_trans_lang")
                if st.button("Translate Entire Document"):
                    with st.spinner("Translating document..."):
                        try:
                            translator = get_translator()
                            translated_doc = translator.translate(st.session_state.extracted_full_text[:5000], dest=doc_trans_lang)
                            st.text_area("Translated Document Preview (First 5000 chars)", value=translated_doc.text, height=250)
                        except Exception as e:
                            st.error(f"Full Document Translation Error: {e}")

            # --- TTS Section ---
            st.markdown("### 🔊 Listen to Extracted Text (TTS)")
            tts_full_lang = st.selectbox("Select Speech Language", ["zh-CN", "en"], index=0, key="full_tts_lang")
            
            if st.button("Generate Audio for Full Document", key="btn_gen_full_audio"):
                with st.spinner("Generating MP3 audio..."):
                    try:
                        full_audio_stream = generate_audio(st.session_state.extracted_full_text, tts_full_lang)
                        audio_data = full_audio_stream.getvalue()
                        
                        st.audio(audio_data, format="audio/mp3")
                        st.download_button(
                            label="📥 Download Full Document Audio (.mp3)",
                            data=audio_data,
                            file_name="full_document_audio.mp3",
                            mime="audio/mp3",
                            key="btn_dl_full_audio"
                        )
                    except Exception as e:
                        st.error(f"Full Document Audio Generation Error: {e}")

            st.divider()

            tab1, tab2 = st.tabs(["📝 Plain Text View", "📊 JSON Structure Preview"])
            with tab1:
                st.text_area(
                    "Full Document Text Result",
                    value=st.session_state.extracted_full_text,
                    height=350,
                    key="full_text_area"
                )
            with tab2:
                json_string = json.dumps(st.session_state.structured_data, ensure_ascii=False, indent=4)
                st.code(json_string, language="json")

    pdf_document.close()
