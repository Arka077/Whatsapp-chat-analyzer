"""
Main Streamlit Application - WhatsApp Chat Analyzer (Modern UI)
"""
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime

# Import your modules
from core import preprocessor
from rag.chat_indexer import ChatIndexer
from utils.date_utils import get_preset_ranges, parse_date
from config.settings import FAISS_INDEX_PATH

# ===================================================================
# AUTO-CREATE REQUIRED FOLDERS (CRITICAL FOR STREAMLIT CLOUD)
# ===================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, INDEX_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MODERN DARK THEME CSS
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0f111a 100%) !important;
        border-right: 1px solid #2d3748 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        color: #25D366 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-align: center !important;
    }
    
    /* Sidebar markdown text */
    [data-testid="stSidebar"] .element-container p {
        color: #a0aec0 !important;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: #2d3748 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Sidebar subheaders */
    [data-testid="stSidebar"] h3 {
        color: #cbd5e0 !important;
        font-size: 1.1rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* File uploader in sidebar */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #2d3748 !important;
        border: 2px dashed #4a5568 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        color: #e2e8f0 !important;
    }
    
    /* Selectbox in sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #2d3748 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        color: #e2e8f0 !important;
    }
    
    /* Buttons in sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #25D366 0%, #128C7E 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(37, 211, 102, 0.2) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 211, 102, 0.35) !important;
    }
    
    /* Success/Error boxes in sidebar */
    [data-testid="stSidebar"] .element-container [data-testid="stNotification"] {
        background-color: #2d3748 !important;
        border-radius: 8px !important;
        border-left: 4px solid #25D366 !important;
    }
    
    /* Metrics in sidebar */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: #2d3748 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #4a5568 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #25D366 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    
    /* Main content area */
    .main {
        background-color: #0f111a !important;
        padding: 2rem !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d29 !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        gap: 0.5rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #a0aec0 !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2d3748 !important;
        color: #25D366 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #25D366 !important;
        color: white !important;
    }
    
    /* Info boxes */
    .element-container [data-testid="stMarkdownContainer"] [data-testid="stNotification"] {
        background: linear-gradient(135deg, #2d3748 0%, #1a1d29 100%) !important;
        border: 1px solid #4a5568 !important;
        border-radius: 12px !important;
        color: #cbd5e0 !important;
    }
    
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #25D366 0%, #128C7E 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(37, 211, 102, 0.3) !important;
    }
    
    /* Regular buttons */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #4a5568 !important;
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        border-color: #25D366 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #25D366 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1d29;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #25D366;
    }
    
    /* Dividers */
    hr {
        border-color: #2d3748 !important;
        margin: 2rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# HELPER: Load chat data from saved index
# ===================================================================
def _load_selected_index_data(selected):
    data_path = os.path.join(FAISS_INDEX_PATH, selected, 'chat_data.pkl')
    if os.path.isfile(data_path):
        try:
            st.session_state.df = pd.read_pickle(data_path)
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to load chat data: {e}")
            return False
    return False

# ===================================================================
# MAIN APP
# ===================================================================
def main():
    st.sidebar.title("💬 WhatsApp Analyzer")
    st.sidebar.markdown("AI-powered chat analysis")
    st.sidebar.markdown("---")

    # Session state init
    for key, default in {
        "df": None, "indexer": None, "is_indexed": False,
        "upload_status": None, "current_page": "Home",
        "index_name": None, "last_uploaded_file_id": None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ===================================================================
    # SAFELY LIST EXISTING INDEXES (NO CRASH IF EMPTY)
    # ===================================================================
    index_dirs = []
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index_dirs = [
                d for d in os.listdir(FAISS_INDEX_PATH)
                if os.path.isdir(os.path.join(FAISS_INDEX_PATH, d))
                and not d.startswith(('.', '__'))
            ]
        except:
            index_dirs = []

    # Load previous index dropdown
    if (not st.session_state.index_name) and index_dirs:
        st.sidebar.markdown("### 📂 Load Previous Chat")
        selected = st.sidebar.selectbox(
            "Choose a saved chat",
            options=[""] + index_dirs,
            format_func=lambda x: "Select a chat..." if not x else x,
            key="load_previous"
        )
        if selected:
            st.session_state.index_name = selected
            index_dir = os.path.join(FAISS_INDEX_PATH, selected)
            has_files = (
                os.path.exists(os.path.join(index_dir, "index.faiss")) and
                os.path.exists(os.path.join(index_dir, "metadata.pkl"))
            )
            loaded = _load_selected_index_data(selected)
            st.session_state.is_indexed = has_files and loaded
            st.rerun()

    # ===================================================================
    # FILE UPLOADER
    # ===================================================================
    st.sidebar.markdown("### 📤 Upload New Chat")
    uploaded_file = st.sidebar.file_uploader(
        "Upload WhatsApp .txt export",
        type=['txt'],
        help="Export chat → Without Media",
        label_visibility="collapsed"
    )

    if uploaded_file and uploaded_file != st.session_state.last_uploaded_file_id:
        st.session_state.last_uploaded_file_id = uploaded_file
        with st.spinner("Processing chat..."):
            try:
                data = uploaded_file.getvalue().decode("utf-8")
                st.session_state.df = preprocessor.preprocess(data)
                st.sidebar.success("✅ Chat loaded successfully!")

                # Generate safe index name
                name = uploaded_file.name.replace(".txt", "").replace(" ", "_")
                safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
                st.session_state.index_name = safe_name
                st.session_state.is_indexed = False
                st.session_state.indexer = None

                # Clean old index
                old_dir = os.path.join(FAISS_INDEX_PATH, safe_name)
                if os.path.exists(old_dir):
                    shutil.rmtree(old_dir, ignore_errors=True)

                # Quick stats
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 📊 Chat Overview")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Messages", len(st.session_state.df))
                with col2:
                    users = st.session_state.df[st.session_state.df['user'] != 'group_notification']['user'].nunique()
                    st.metric("Users", users)

            except Exception as e:
                st.sidebar.error(f"❌ Upload failed: {str(e)[:100]}")

    # ===================================================================
    # INDEXING SECTION
    # ===================================================================
    if st.session_state.df is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔄 Vector Indexing")

        if not st.session_state.is_indexed:
            if st.sidebar.button("🚀 Index Chat for AI Search", type="primary", use_container_width=True):
                with st.spinner("⏳ Indexing messages... This takes 10-60 seconds"):
                    try:
                        indexer = ChatIndexer(index_name=st.session_state.index_name)
                        success = indexer.index_chat_data(st.session_state.df)
                        if success:
                            os.makedirs(os.path.join(FAISS_INDEX_PATH, st.session_state.index_name), exist_ok=True)
                            st.session_state.df.to_pickle(
                                os.path.join(FAISS_INDEX_PATH, st.session_state.index_name, "chat_data.pkl")
                            )
                            st.session_state.indexer = indexer
                            st.session_state.is_indexed = True
                            st.sidebar.success("✅ Indexing complete!")
                            st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"❌ Indexing failed: {e}")
        else:
            st.sidebar.success("✅ Chat indexed & ready")
            if not st.session_state.indexer:
                try:
                    st.session_state.indexer = ChatIndexer(index_name=st.session_state.index_name)
                except:
                    pass

    # ===================================================================
    # NAVIGATION
    # ===================================================================
    if st.session_state.df is not None and st.session_state.is_indexed:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 Navigation")
        pages = ["Home", "Analytics", "Q&A", "Sentiment", "Topics"]
        icons = ["🏠", "📊", "💬", "😊", "📚"]

        for page, icon in zip(pages, icons):
            if st.sidebar.button(f"{icon} {page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()

    # ===================================================================
    # MAIN CONTENT
    # ===================================================================
    if st.session_state.df is None:
        st.title("💬 WhatsApp Chat Analyzer")
        st.markdown("### Upload your WhatsApp chat export (.txt) to begin")
        st.info("📱 Go to WhatsApp → Chat → More → Export Chat → Without Media")
        return

    if not st.session_state.is_indexed:
        st.info("✨ Chat uploaded! Click **'Index Chat for AI Search'** in the sidebar to unlock all features.")
        return

    page = st.session_state.current_page

    if page in ["Home", "Analytics"]:
        st.header("📊 Chat Analytics")
        from ui.page_modules import analytics_page
        analytics_page.show(st.session_state.df)

    elif page == "Q&A":
        st.header("💬 Ask Anything About Your Chat")
        from ui.page_modules import rag_qa_page
        rag_qa_page.show(st.session_state.df)

    elif page == "Sentiment":
        st.header("😊 Sentiment Analysis")
        from ui.page_modules import sentiment_page
        sentiment_page.show(st.session_state.df)

    elif page == "Topics":
        st.header("📚 Topic Discovery")
        from ui.page_modules import topics_page
        topics_page.show(st.session_state.df)

if __name__ == "__main__":
    main()