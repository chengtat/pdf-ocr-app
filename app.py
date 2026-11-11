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

# --- Custom Modern UI Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            border-radius: 6px;
            font-weight: 500;
        }
        /* 關鍵字高亮樣式 */
        mark {
            background-color: #ffeb3b;
            color: #000000;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        .highlight-box {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 20px;
            border-radius: 8px;
            max-height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: monospace;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Multilingual PDF OCR Studio Pro")
st.caption("Advanced OCR, Table Extraction, Bounding Box Inspector, Translation & Text-to-Speech Engine")

# --- Session State Initialization ---
default_states = {
    "extracted_full_text": "",
    "structured_data": [],
    "total_chinese_chars": 0,
    "total_english_words": 0,
    "single_page_results": {},
    "page_tables": {},
    "translated_full_text": "",
    "last_uploaded_filename": None
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
    if engine_type == "RapidOCR":
        ocr_engine = load_rapidocr_engine()
        result, _ = ocr_engine(img_np)
        return result
    else:
        reader = load_easyocr_engine()
        raw_result = reader.readtext(img_np)
        formatted_result = []
        for box, text, conf in raw_result:
            formatted_box = [[int(pt[0]), int(pt[1])] for pt in box]
            formatted_result.append([formatted_box, text, conf])
        return formatted_result

def process_page(page, dpi=150, engine_type="RapidOCR"):
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)
    result = process_ocr(img_np, engine_type)
    pix = None
    return img_np, result

def extract_tables_from_pdf(pdf_bytes, page_idx):
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

def highlight_text(text, query):
    """將文字中的搜尋關鍵字用 <mark> 標籤包覆以實現高亮"""
    if not query.strip() or not text:
        return text
    # 使用 re.sub 進行不分大小寫的替換
    escaped_query = re.escape(query)
    highlighted = re.sub(
        f"({escaped_query})",
        r"<mark>\1</mark>",
        text,
        flags=re.IGNORECASE
    )
    return highlighted

# --- Sidebar Configuration Hub ---
with st.sidebar:
    st.header("⚙️ Studio Control Panel")
    
    with st.container(border=True):
        st.subheader("OCR Engine Setup")
        engine_choice = st.selectbox("Backend Engine", ["RapidOCR", "EasyOCR"])
        dpi_setting = st.slider("Render Resolution (DPI)", 100, 300, 150, 50)

    with st.container(border=True):
        st.subheader("Visual Options")
        show_bounding_boxes = st.checkbox("Draw Bounding Boxes", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)

    with st.container(border=True):
        st.subheader("🔍 Keyword Search")
        search_query = st.text_input("Find in document:", placeholder="Type keyword...")

    st.markdown("---")
    st.info("💡 **Tip:** Upload your multi-page bilingual PDF below to initialize the workspace.")

# --- Main File Uploader Area ---
uploaded_file = st.file_uploader("📂 Upload PDF Document", type=["pdf"])

if uploaded_file is not None:
    # 檢查是否為新上傳的檔案，若是則清空舊狀態
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.extracted_full_text = ""
        st.session_state.structured_data = []
        st.session_state.total_chinese_chars = 0
        st.session_state.total_english_words = 0
        st.session_state.single_page_results = {}
        st.session_state.page_tables = {}
        st.session_state.translated_full_text = ""
        st.session_state.last_uploaded_filename = uploaded_file.name

    pdf_bytes = uploaded_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)

    # Top Status Bar Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 File Name", uploaded_file.name)
    col2.metric("📊 Total Pages", total_pages)
    col3.metric("⚡ Active Engine", engine_choice)
    col4.metric("Resolution", f"{dpi_setting} DPI")

    st.divider()

    # --- Full Document Pre-extraction Trigger ---
    if not st.session_state.extracted_full_text:
        with st.container(border=True):
            st.subheader("🚀 Batch Processing")
            st.write("Click below to run full document scanning across all pages instantly.")
            if st.button("Start Full Document OCR Extraction", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                temp_full_text = ""
                temp_structured_data = []
                temp_chinese_chars = 0
                temp_english_words = 0
                
                for page_idx in range(total_pages):
                    status_text.text(f"Processing page {page_idx + 1} of {total_pages}...")
                    page = pdf_document[page_idx]
                    img_np, result = process_page(page, dpi=dpi_setting, engine_type=engine_choice)
                    
                    page_lines = []
                    page_text = ""
                    if result:
                        page_text = "\n".join([line[1] for line in result])
                        for line in result:
                            text_val = line[1]
                            temp_chinese_chars += sum(1 for c in text_val if '\u4e00' <= c <= '\u9fff')
                            temp_english_words += len([w for w in text_val.split() if w.isascii() and w.isalpha()])
                            if show_confidence:
                                page_lines.append({"text": text_val, "confidence": float(line[2])})
                    
                    temp_full_text += f"\n\n--- Page {page_idx + 1} ---\n\n" + page_text
                    temp_structured_data.append({"page": page_idx + 1, "content": page_text, "lines": page_lines})
                    st.session_state.page_tables[page_idx] = extract_tables_from_pdf(pdf_bytes, page_idx)
                    
                    del img_np, result
                    gc.collect()
                    progress_bar.progress((page_idx + 1) / total_pages)
                
                st.session_state.extracted_full_text = temp_full_text
                st.session_state.structured_data = temp_structured_data
                st.session_state.total_chinese_chars = temp_chinese_chars
                st.session_state.total_english_words = temp_english_words
                status_text.text("Extraction complete!")
                st.success("Document scanned successfully!")
                st.rerun()

    # --- Modern Multi-Tab Studio Workspace ---
    tab_overview, tab_preview, tab_tables, tab_translate = st.tabs([
        "📑 Document Overview & Export",
        "🔍 Page Studio & Bounding Boxes",
        "📊 Table Data Analytics",
        "🌐 AI Translation & Audio Studio"
    ])

    # --- TAB 1: OVERVIEW & EXPORT ---
    with tab_overview:
        st.markdown("### Document Analytics & Full Export")
        if st.session_state.extracted_full_text:
            if search_query.strip():
                matches = len(re.findall(re.escape(search_query), st.session_state.extracted_full_text, re.IGNORECASE))
                st.success(f"🔍 Found **{matches}** match(es) for **'{search_query}'** in the full document.")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Characters", len(st.session_state.extracted_full_text))
            col_m2.metric("Chinese Characters", st.session_state.total_chinese_chars)
            col_m3.metric("English Words", st.session_state.total_english_words)

            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="📄 Download Full Text (.txt)",
                    data=st.session_state.extracted_full_text,
                    file_name="extracted_ocr_text.txt",
                    mime="text/plain",
                    type="primary",
                    use_container_width=True
                )
            with col_d2:
                json_string = json.dumps(st.session_state.structured_data, ensure_ascii=False, indent=4)
                st.download_button(
                    label="📊 Download Structured Data (.json)",
                    data=json_string,
                    file_name="extracted_ocr_data.json",
                    mime="application/json",
                    use_container_width=True
                )

            st.markdown("### 📝 Full Text View (with Keyword Highlighting)")
            
            # 處理並渲染高亮文本
            highlighted_html = highlight_text(st.session_state.extracted_full_text, search_query)
            st.markdown(f'<div class="highlight-box">{highlighted_html}</div>', unsafe_allow_html=True)
        else:
            st.info("Run the batch extraction above to populate the document overview.")

    # --- TAB 2: PAGE STUDIO & BOUNDING BOXES ---
    with tab_preview:
        st.markdown("### Single Page Visual Inspector & Line Corrector")
        page_num = st.slider("Select Page to Inspect", 1, total_pages, 1, key="preview_page_slider") - 1
        page = pdf_document[page_num]

        col_img, col_txt = st.columns(2, gap="medium")

        with col_img:
            with st.container(border=True):
                st.subheader(f"Page {page_num + 1} Visual Canvas")
                run_page_btn = st.button("🔍 Run Page OCR", type="primary", key=f"btn_run_page_{page_num}")
                
                if run_page_btn or page_num in st.session_state.single_page_results:
                    if run_page_btn or page_num not in st.session_state.single_page_results:
                        with st.spinner("Processing page layout..."):
                            img_np, result = process_page(page, dpi=dpi_setting, engine_type=engine_choice)
                            st.session_state.single_page_results[page_num] = {"img_np": img_np, "result": result}
                            if page_num not in st.session_state.page_tables:
                                st.session_state.page_tables[page_num] = extract_tables_from_pdf(pdf_bytes, page_num)
                    
                    cached = st.session_state.single_page_results[page_num]
                    img_np, result = cached["img_np"], cached["result"]
                    
                    if result and show_bounding_boxes:
                        annotated_img = img_np.copy()
                        for idx, line in enumerate(result):
                            box = np.array(line[0], dtype=np.int32)
                            cv2.polylines(annotated_img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                            cv2.putText(annotated_img, str(idx), (int(box[0][0]), max(15, int(box[0][1]) - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        st.image(annotated_img, use_container_width=True)
                    else:
                        st.image(img_np, use_container_width=True)
                else:
                    pix = page.get_pixmap(dpi=100)
                    st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)

        with col_txt:
            with st.container(border=True):
                st.subheader("Extracted Line Editor")
                if page_num in st.session_state.single_page_results:
                    result = st.session_state.single_page_results[page_num]["result"]
                    if result:
                        page_text = "\n".join([line[1] for line in result])
                        st.text_area("Page Text", value=page_text, height=180, key=f"page_txt_area_{page_num}")

                        with st.expander("🛠️ Interactive Line Bounding Box Corrector"):
                            line_options = [f"[{i}] {line[1][:30]}... (Conf: {float(line[2]):.2f})" for i, line in enumerate(result)]
                            selected_line_idx = st.selectbox("Select Line Index:", range(len(result)), format_func=lambda x: line_options[x], key=f"sel_line_{page_num}")
                            
                            target_line = result[selected_line_idx]
                            corrected_input = st.text_input("Edit Line Text:", value=target_line[1], key=f"edit_val_{page_num}_{selected_line_idx}")
                            if st.button("Save Correction", key=f"save_edit_{page_num}_{selected_line_idx}"):
                                st.session_state.single_page_results[page_num]["result"][selected_line_idx][1] = corrected_input
                                st.success("Updated successfully!")
                                st.rerun()

                        if show_confidence:
                            with st.expander("🔍 View Line Confidence Scores"):
                                for line in result:
                                    st.write(f"- `{line[1]}` — **{float(line[2]):.4f}**")
                    else:
                        st.info("No text detected on this page.")
                else:
                    st.info("Click 'Run Page OCR' to scan this specific page.")

    # --- TAB 3: TABLE DATA ANALYTICS ---
    with tab_tables:
        st.markdown("### Auto-Detected Tables & CSV Export")
        table_page_sel = st.selectbox("Select Page for Table Inspection", range(total_pages), format_func=lambda x: f"Page {x + 1}")
        
        if table_page_sel not in st.session_state.page_tables:
            st.session_state.page_tables[table_page_sel] = extract_tables_from_pdf(pdf_bytes, table_page_sel)
            
        tables = st.session_state.page_tables.get(table_page_sel, [])
        if tables:
            st.success(f"Found **{len(tables)}** table(s) on Page {table_page_sel + 1}.")
            for t_idx, df in enumerate(tables):
                with st.container(border=True):
                    st.write(f"**Table #{t_idx + 1}**")
                    st.dataframe(df, use_container_width=True)
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download Table #{t_idx + 1} as CSV",
                        data=csv_bytes,
                        file_name=f"page_{table_page_sel + 1}_table_{t_idx + 1}.csv",
                        mime="text/csv",
                        key=f"dl_tbl_{table_page_sel}_{t_idx}"
                    )
        else:
            st.info(f"No structured tables detected on Page {table_page_sel + 1}.")

    # --- TAB 4: AI TRANSLATION & AUDIO STUDIO ---
    with tab_translate:
        st.markdown("### 🌐 AI Translation & Audio Studio")
        st.write("Easily translate or generate audio directly from your extracted text using the selector below.")

        if not st.session_state.extracted_full_text:
            st.warning("Please run Full Document OCR extraction first or scan a page to use the Studio.")
        else:
            source_choice = st.radio(
                "Choose Source Content to Process:",
                ["Entire Extracted Document", "Specific Page Text"],
                horizontal=True
            )

            if source_choice == "Entire Extracted Document":
                active_text = st.session_state.extracted_full_text
            else:
                target_page_idx = st.selectbox("Select Page Source", range(total_pages), format_func=lambda x: f"Page {x + 1}")
                if target_page_idx in st.session_state.single_page_results:
                    res_lines = st.session_state.single_page_results[target_page_idx]["result"]
                    active_text = "\n".join([line[1] for line in res_lines]) if res_lines else ""
                else:
                    active_text = "Page has not been scanned yet. Run OCR in the Page Studio tab or run Full Extraction."

            col_trans, col_tts = st.columns(2, gap="medium")

            with col_trans:
                with st.container(border=True):
                    st.subheader("🌐 Machine Translation")
                    st.text_area("Source Text Preview", value=active_text[:1000] + ("..." if len(active_text) > 1000 else ""), height=150, disabled=True)
                    
                    target_lang = st.selectbox("Target Language", ["zh-cn", "zh-tw", "en", "ja", "fr", "de", "es"], key="studio_target_lang")
                    
                    if st.button("🚀 Translate Selected Content", type="primary", key="btn_exec_trans"):
                        with st.spinner("Translating via Google Translate API..."):
                            try:
                                translator = get_translator()
                                max_len = 4500
                                text_chunks = [active_text[i:i + max_len] for i in range(0, len(active_text), max_len)]
                                translated_result = ""
                                for chunk in text_chunks:
                                    res = translator.translate(chunk, dest=target_lang)
                                    translated_result += res.text + "\n"
                                
                                st.session_state.translated_full_text = translated_result
                                st.success("Translation complete!")
                            except Exception as e:
                                st.error(f"Translation Error: {e}")

                    if st.session_state.translated_full_text:
                        st.markdown("**Translated Result:**")
                        st.text_area("Translated Output", value=st.session_state.translated_full_text, height=200, key="trans_output_box")
                        st.download_button(
                            label="📥 Download Translated Text (.txt)",
                            data=st.session_state.translated_full_text,
                            file_name="translated_output.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

            with col_tts:
                with st.container(border=True):
                    st.subheader("🔊 Audio Synthesis (TTS)")
                    tts_source_mode = st.radio("Synthesize Voice From:", ["Original Source Text", "Translated Text Result"], horizontal=True, key="tts_src_mode")
                    
                    tts_active_text = active_text if tts_source_mode == "Original Source Text" else st.session_state.translated_full_text
                    
                    st.text_area("Voice Input Preview", value=tts_active_text[:1000] + ("..." if len(tts_active_text) > 1000 else ""), height=150, disabled=True)
                    
                    tts_lang = st.selectbox("Speech Language", ["zh-CN", "en"], key="studio_tts_lang")
                    
                    if st.button("🎙️ Generate MP3 Audio", type="primary", key="btn_exec_tts"):
                        if not tts_active_text.strip() or "not been scanned" in tts_active_text:
                            st.error("Selected text is empty or invalid for speech synthesis.")
                        else:
                            with st.spinner("Synthesizing speech audio..."):
                                try:
                                    audio_stream = generate_audio(tts_active_text, tts_lang)
                                    audio_bytes = audio_stream.getvalue()
                                    
                                    st.audio(audio_bytes, format="audio/mp3")
                                    st.download_button(
                                        label="📥 Download Generated Audio (.mp3)",
                                        data=audio_bytes,
                                        file_name="studio_audio.mp3",
                                        mime="audio/mp3",
                                        use_container_width=True,
                                        key="btn_dl_studio_audio"
                                    )
                                except Exception as e:
                                    st.error(f"TTS Audio Generation Error: {e}")

    pdf_document.close()
