import streamlit as st
import fitz   # PyMuPDF to read PDFs
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import numpy as np
import json
import cv2
import asyncio
import edge_tts
import io

# Page configuration
st.set_page_config(page_title="Multilingual PDF OCR Studio + Edge-TTS", layout="wide")

st.title("📄 Multilingual PDF OCR Studio + Edge-TTS")
st.write("Upload a PDF document to extract text using RapidOCR and convert to high-quality speech (Supports Cantonese, Mandarin, and English).")

# Initialize Session State variables to persist results across reruns
if "extracted_full_text" not in st.session_state:
    st.session_state.extracted_full_text = ""
if "structured_data" not in st.session_state:
    st.session_state.structured_data = []
if "total_chinese_chars" not in st.session_state:
    st.session_state.total_chinese_chars = 0
if "total_english_words" not in st.session_state:
    st.session_state.total_english_words = 0

# Sidebar Controls for Advanced Features
st.sidebar.header("⚙️ Processing Settings")
mode = st.sidebar.radio("Extraction Mode", ["Full Document", "Single Page Preview"])
show_bounding_boxes = st.sidebar.checkbox("Draw Bounding Boxes on Image", value=True)
show_confidence = st.sidebar.checkbox("Show Text Confidence Scores", value=False)

# Edge-TTS Voice Mapping Dictionary
voice_options = {
    "🇭🇰 廣東話 (女聲 - HiuMaan)": "zh-HK-HiuMaanNeural",
    "🇭🇰 廣東話 (男聲 - WanLung)": "zh-HK-WanLungNeural",
    "🇨🇳 普通話 (女聲 - Xiaoxiao)": "zh-CN-XiaoxiaoNeural",
    "🇺🇸 English (Aria)": "en-US-AriaNeural",
}

# Initialize lightweight RapidOCR engine (cached)
@st.cache_resource
def load_ocr_engine():
    return RapidOCR()

with st.spinner("Loading OCR engine..."):
    ocr_engine = load_ocr_engine()

# Async helper function for Edge-TTS streaming
async def generate_edge_audio(text, voice_name):
    communicate = edge_tts.Communicate(text, voice_name)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)

    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", total_pages)
    col2.metric("File Name", uploaded_file.name)
    col3.metric("Engine", "RapidOCR + Edge-TTS")

    st.divider()

    if mode == "Single Page Preview":
        page_num = st.sidebar.slider("Select Page to Preview", 1, total_pages, 1) - 1
        page = pdf_document[page_num]
        
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)
        
        col_img, col_txt = st.columns(2)
        with col_img:
            st.subheader(f"Page {page_num + 1} Preview")
            
            if st.button("Run OCR & Draw Boxes"):
                with st.spinner("Running OCR..."):
                    result, _ = ocr_engine(img_np)
                    if result and show_bounding_boxes:
                        annotated_img = img_np.copy()
                        for line in result:
                            box = np.array(line[0], dtype=np.int32)
                            cv2.polylines(annotated_img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                        st.image(annotated_img, use_container_width=True)
                    else:
                        st.image(img, use_container_width=True)
            else:
                st.image(img, use_container_width=True)
            
        with col_txt:
            st.subheader("Extracted Text & Speech")
            if st.button("Extract Text Only"):
                with st.spinner("Running OCR..."):
                    result, _ = ocr_engine(img_np)
                    if result:
                        page_text = "\n".join([line[1] for line in result])
                        st.text_area("Result", value=page_text, height=250)
                        
                        # --- 🔊 Edge-TTS 語音整合 ---
                        st.markdown("### 🔊 Edge-TTS Audio Playback")
                        selected_label = st.selectbox(
                            "Select Voice", 
                            options=list(voice_options.keys()), 
                            index=0, 
                            key=f"voice_{page_num}"
                        )
                        voice_id = voice_options[selected_label]
                        
                        if st.button("Generate Speech", key=f"gen_audio_{page_num}"):
                            with st.spinner("Synthesizing audio..."):
                                try:
                                    audio_data = asyncio.run(generate_edge_audio(page_text, voice_id))
                                    audio_fp = io.BytesIO(audio_data)
                                    
                                    st.audio(audio_fp, format="audio/mp3")
                                    st.download_button(
                                        label="📥 Download Audio (.mp3)",
                                        data=audio_fp,
                                        file_name=f"page_{page_num + 1}_audio.mp3",
                                        mime="audio/mp3",
                                        key=f"download_audio_{page_num}"
                                    )
                                except Exception as e:
                                    st.error(f"TTS Error: {e}")
                            
                        st.download_button(
                            label="📥 Download Page Text as .txt",
                            data=page_text,
                            file_name=f"page_{page_num + 1}_text.txt",
                            mime="text/plain",
                            key=f"download_txt_{page_num}"
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
            
            temp_full_text = ""
            temp_structured_data = []
            temp_chinese_chars = 0
            temp_english_words = 0
            
            for page_num in range(total_pages):
                status_text.text(f"Processing page {page_num + 1} of {total_pages}...")
                page = pdf_document[page_num]
                
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                
                result, _ = ocr_engine(img_np)
                
                page_lines = []
                page_text = ""
                if result:
                    page_text = "\n".join([line[1] for line in result])
                    for line in result:
                        text_val = line[1]
                        temp_chinese_chars += sum(1 for c in text_val if '\u4e00' <= c <= '\u9fff')
                        temp_english_words += len([w for w in text_val.split() if w.isascii()])
                        
                        if show_confidence:
                            page_lines.append({"text": text_val, "confidence": float(line[2])})
                
                temp_full_text += f"\n\n--- Page {page_num + 1} ---\n\n" + page_text
                temp_structured_data.append({"page": page_num + 1, "content": page_text, "lines": page_lines})
                
                progress_bar.progress((page_num + 1) / total_pages)
                
            st.session_state.extracted_full_text = temp_full_text
            st.session_state.structured_data = temp_structured_data
            st.session_state.total_chinese_chars = temp_chinese_chars
            st.session_state.total_english_words = temp_english_words
            
            status_text.text("OCR Processing Complete!")
            st.success("All pages successfully processed!")

        if st.session_state.extracted_full_text:
            with st.expander("📊 Document Text Analytics", expanded=True):
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                stat_col1.metric("Total Characters", len(st.session_state.extracted_full_text))
                stat_col2.metric("Estimated Chinese Characters", st.session_state.total_chinese_chars)
                stat_col3.metric("Estimated English Words", st.session_state.total_english_words)

            st.markdown("### 📥 Export Options & Edge-TTS")
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
            
            # --- 全文 Edge-TTS 語音轉檔 ---
            st.markdown("### 🔊 Listen to Full Document (Edge-TTS)")
            selected_full_label = st.selectbox(
                "Select Voice for Full Doc",
                options=list(voice_options.keys()),
                index=0
            )
            voice_full_id = voice_options[selected_full_label]
            
            if st.button("Generate Audio for Full Document"):
                with st.spinner("Synthesizing full document audio..."):
                    try:
                        audio_data = asyncio.run(generate_edge_audio(st.session_state.extracted_full_text, voice_full_id))
                        full_audio_fp = io.BytesIO(audio_data)
                        
                        st.audio(full_audio_fp, format="audio/mp3")
                        st.download_button(
                            label="📥 Download Full Document Audio (.mp3)",
                            data=full_audio_fp,
                            file_name="full_document_audio.mp3",
                            mime="audio/mp3"
                        )
                    except Exception as e:
                        st.error(f"Full TTS Error: {e}")

            st.divider()

            tab1, tab2 = st.tabs(["📝 Plain Text View", "📊 JSON Structure Preview"])
            with tab1:
                st.text_area("Full Document Result", value=st.session_state.extracted_full_text, height=350, key="full_text_area")
            with tab2:
                st.code(json_string, language="json")
