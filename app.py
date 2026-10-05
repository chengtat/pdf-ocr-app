import streamlit as st
from gtts import gTTS
import tempfile
import os
from pypdf import PdfReader

# 頁面基本設定
st.set_page_config(
    page_title="PDF OCR & Cantonese TTS App",
    page_icon="🔊",
    layout="centered"
)

st.title("📄 PDF 轉文字與廣東話語音系統")
st.markdown("上傳 PDF 檔案，系統會提取頁面文字，並透過 **gTTS (Google 廣東話)** 進行語音合成，絕不發生連線逾時！")

# 1. 檔案上傳
uploaded_file = st.file_uploader("請上傳你的 PDF 檔案", type=["pdf"])

if uploaded_file is not None:
    try:
        # 讀取 PDF
        reader = PdfReader(uploaded_file)
        total_pages = len(reader.pages)
        st.success(f"成功載入 PDF！總共有 {total_pages} 頁。")
        
        # 頁面選擇器
        page_num = st.selectbox(
            "選擇要檢視及轉換的頁面：",
            range(total_pages),
            format_func=lambda x: f"第 {x + 1} 頁"
        )
        
        # 提取指定頁面的文字
        page = reader.pages[page_num]
        page_text = page.extract_text()
        
        st.subheader(f"📖 第 {page_num + 1} 頁提取文字：")
        
        if page_text and page_text.strip():
            # 顯示提取出的文字編輯框
            edited_text = st.text_area(
                "文字內容（可手動修改後再轉語音）：",
                value=page_text,
                height=220,
                key=f"text_area_{page_num}"
            )
            
            # 2. 廣東話語音生成按鈕
            if st.button("🔊 生成廣東話語音 (Cantonese TTS)", key=f"tts_btn_{page_num}", type="primary"):
                with st.spinner("正在透過 Google 服務合成廣東話語音..."):
                    try:
                        # 限制字數以防過長 (gTTS 建議單次不要過長)
                        clean_text = edited_text[:1200] if len(edited_text) > 1200 else edited_text
                        
                        if not clean_text.strip():
                            st.warning("文字內容為空，無法生成語音。")
                        else:
                            # 關鍵：lang="zh-yue" 指定為廣東話
                            tts = gTTS(text=clean_text, lang="zh-yue", slow=False)
                            
                            # 建立暫存檔儲存 MP3
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                tmp_path = tmp_file.name
                                
                            tts.save(tmp_path)
                            
                            # 讀取二進位資料
                            with open(tmp_path, "rb") as f:
                                audio_bytes = f.read()
                                
                            # 刪除暫存檔
                            os.unlink(tmp_path)
                            
                            st.success("✅ 語音生成成功！")
                            
                            # 播放器
                            st.audio(audio_bytes, format="audio/mp3")
                            
                            # 下載按鈕
                            st.download_button(
                                label="📥 下載廣東話音頻 (.mp3)",
                                data=audio_bytes,
                                file_name=f"page_{page_num + 1}_cantonese.mp3",
                                mime="audio/mp3",
                                key=f"download_audio_{page_num}"
                            )
                    except Exception as e:
                        st.error(f"TTS Error: {e}")
        else:
            st.info("⚠️ 這頁偵測唔到文字（可能係圖片型 PDF，需配合 OCR 辨識模組）。")
            
    except Exception as e:
        st.error(f"讀取 PDF 發生錯誤: {e}")
