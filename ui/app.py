"""
Main Streamlit Application - Multilingual WhatsApp Analyzer
"""
import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import os
from core import preprocessor
from rag.chat_indexer import ChatIndexer
from utils.date_utils import get_preset_ranges, parse_date
from datetime import datetime
from config.settings import FAISS_INDEX_PATH
from pathlib import Path

# === AUTO-CREATE REQUIRED FOLDERS ON STARTUP ===
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, INDEX_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
# Page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
st.markdown("""
<style>
    /* Remove sidebar white background - use theme color instead */
    .st-emotion-cache-1r6muj3 {
        background-color: #f0f2f6 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    /* Main container styling */
    .main {
        padding: 0rem 0rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    
    /* Alert boxes styling */
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Metric styling */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
    }
    
    /* Divider */
    .stDivider {
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1a1a1a;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# When a previous chat index is loaded from sidebar, also load its chat_data.pkl for analysis
def _load_selected_index_data(selected):
    data_path = os.path.join(FAISS_INDEX_PATH, selected, 'chat_data.pkl')
    if os.path.isfile(data_path):
        st.session_state.df = pd.read_pickle(data_path)
        return True
    else:
        st.session_state.df = None
        st.session_state.is_indexed = False
        st.sidebar.error(f"No saved chat data found in {data_path}. Please re-upload and re-index this chat.")
        return False

def main():
    """Main application function"""
    st.sidebar.title("💬 WhatsApp Analyzer")
    st.sidebar.markdown("Advanced conversation analysis with RAG, sentiment, and topics")
    st.sidebar.markdown("---")
    
    # Helper function to check if index exists
    def is_index_available():
        index_name = st.session_state.get('index_name')
        if index_name:
            index_dir = os.path.join(FAISS_INDEX_PATH, str(index_name).replace(' ', '_'))
            index_file = os.path.join(index_dir, "index.faiss")
            metadata_file = os.path.join(index_dir, "metadata.pkl")
            return os.path.exists(index_file) and os.path.exists(metadata_file)

        # Fallback to default index
        index_file = os.path.join(FAISS_INDEX_PATH, "default", "index.faiss")
        metadata_file = os.path.join(FAISS_INDEX_PATH, "default", "metadata.pkl")
        return os.path.exists(index_file) and os.path.exists(metadata_file)
    
    # Session state initialization
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'indexer' not in st.session_state:
        st.session_state.indexer = None
    if 'is_indexed' not in st.session_state:
        st.session_state.is_indexed = False
    if 'upload_status' not in st.session_state:
        st.session_state.upload_status = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    if 'index_name' not in st.session_state:
        st.session_state.index_name = None
    if 'last_uploaded_file_id' not in st.session_state:
        st.session_state.last_uploaded_file_id = None
    
    # On new session, or if index_name not set, list available indices
    index_dirs = [d for d in os.listdir(FAISS_INDEX_PATH) if os.path.isdir(os.path.join(FAISS_INDEX_PATH, d))]
    if ('index_name' not in st.session_state or not st.session_state.index_name) and index_dirs:
        selected = st.sidebar.selectbox("📂 Load previous chat index", index_dirs, key="existing_index_select")
        if selected:
            st.session_state.index_name = selected
            available = os.path.exists(os.path.join(FAISS_INDEX_PATH, selected, 'index.faiss')) and os.path.exists(os.path.join(FAISS_INDEX_PATH, selected, 'metadata.pkl'))
            loaded_data = _load_selected_index_data(selected)
            st.session_state.is_indexed = available and loaded_data
            st.rerun()

    # File upload handling
    uploaded_file = st.sidebar.file_uploader(
        "Select WhatsApp export",
        type=['txt'],
        help="Export without media using WhatsApp settings",
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    # Process uploaded file only if it's new
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.file_id}"
        
        # Only process if this is a NEW file
        if file_id != st.session_state.last_uploaded_file_id:
            try:
                bytes_data = uploaded_file.getvalue()
                data = bytes_data.decode("utf-8")
                
                st.session_state.df = preprocessor.preprocess(data)
                st.session_state.upload_status = "success"
                st.sidebar.success("✅ Chat processed successfully!")
                
                # Prepare index name
                filename = getattr(uploaded_file, 'name', None) or "chat"
                index_name = os.path.splitext(filename)[0]
                safe_index_name = str(index_name).replace(' ', '_').replace('\\', '_').replace('/', '_').replace(':', '_')
                st.session_state.index_name = safe_index_name
                st.session_state.is_indexed = False
                st.session_state.indexer = None
                st.session_state.last_uploaded_file_id = file_id
                
                # Remove any existing index for this chat
                index_dir = os.path.join(FAISS_INDEX_PATH, safe_index_name)
                if os.path.exists(index_dir):
                    try:
                        shutil.rmtree(index_dir)
                    except Exception:
                        pass

                # Show basic stats
                st.sidebar.markdown("---")
                st.sidebar.subheader("📊 Chat Overview")
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Messages", len(st.session_state.df))
                with col2:
                    unique_users = st.session_state.df[st.session_state.df['user'] != 'group_notification']['user'].nunique()
                    st.metric("Users", unique_users)
                
                date_min = st.session_state.df['only_date'].min()
                date_max = st.session_state.df['only_date'].max()
                st.sidebar.metric("Duration", f"{(date_max - date_min).days} days")
                
            except Exception as e:
                st.session_state.upload_status = "error"
                st.sidebar.error(f"❌ Upload failed: {str(e)[:100]}")
                st.sidebar.info("💡 Make sure you're uploading a valid WhatsApp export file")
    
    # Show indexing section if chat data exists
    if st.session_state.df is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Vector Indexing")
        st.sidebar.markdown("<p style='font-size: 0.85rem; color: #666;'>Index your chat for RAG search and analysis.</p>", unsafe_allow_html=True)
        
        if not st.session_state.is_indexed:
            if st.sidebar.button("🚀 Index Chat Data", width='stretch', key="index_btn"):
                with st.spinner("⏳ Indexing messages..."):
                    try:
                        index_name_to_use = st.session_state.get('index_name')
                        st.session_state.indexer = ChatIndexer(index_name=index_name_to_use)
                        st.sidebar.info(f"📝 Using index name: {index_name_to_use}")
                        st.sidebar.info(f"📂 Index dir: {st.session_state.indexer.vector_store.index_dir}")
                        st.sidebar.info(f"📄 Index path: {st.session_state.indexer.vector_store.index_path}")
                        
                        success = st.session_state.indexer.index_chat_data(st.session_state.df)
                        st.sidebar.info(f"📊 Indexing returned: {success}")
                        
                        if success:
                            data_path = os.path.join(FAISS_INDEX_PATH, index_name_to_use, 'chat_data.pkl')
                            st.sidebar.info(f"💾 Saving to: {data_path}")
                            try:
                                st.session_state.df.to_pickle(data_path)
                                st.sidebar.success("✅ Chat data saved!")
                            except Exception as save_e:
                                st.sidebar.warning(f"Could not save chat data: {save_e}")
                            
                            st.session_state.is_indexed = True
                            st.sidebar.success("✅ Indexing complete!")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Indexing failed - index_chat_data returned False")
                    except Exception as e:
                        import traceback
                        st.sidebar.error(f"❌ Error during indexing: {str(e)}")
                        st.sidebar.error(traceback.format_exc())
        else:
            st.sidebar.success("✅ Chat indexed and ready")
            if not st.session_state.indexer and is_index_available():
                try:
                    st.session_state.indexer = ChatIndexer(index_name=st.session_state.get('index_name'))
                except Exception:
                    st.session_state.indexer = None

            if st.session_state.indexer:
                try:
                    stats = st.session_state.indexer.get_index_stats()
                    st.sidebar.info(f"📌 {stats['total_vectors']} messages indexed")
                except:
                    st.sidebar.info("📌 Chat data indexed")
    
    # Debug info sidebar
    st.sidebar.markdown('---')
    st.sidebar.info(f"Current index: {st.session_state.get('index_name', 'None')}")
    st.sidebar.info(f"Indexed? {'✅' if st.session_state.get('is_indexed') else '❌'}")
    
    with st.sidebar.expander("🐛 Debug Info"):
        st.write(f"df exists: {st.session_state.df is not None}")
        st.write(f"is_indexed: {st.session_state.is_indexed}")
        st.write(f"index_name: {st.session_state.index_name}")
        if st.session_state.df is not None:
            st.write(f"df shape: {st.session_state.df.shape}")
        index_name = st.session_state.get('index_name')
        if index_name:
            index_dir = os.path.join(FAISS_INDEX_PATH, str(index_name).replace(' ', '_'))
            st.write(f"Index dir: {index_dir}")
            st.write(f"Index dir exists: {os.path.exists(index_dir)}")
            if os.path.exists(index_dir):
                st.write(f"Files in dir: {os.listdir(index_dir)}")

    # Navigation Menu
    if st.session_state.df is not None and st.session_state.is_indexed:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📑 Navigation")

        nav_options = ["Home", "📊 Analytics", "❓ Q&A", "💭 Sentiment", "📚 Topics"]

        for option in nav_options:
            if st.sidebar.button(option, width='stretch', key=f"nav_{option}"):
                st.session_state.current_page = option
    
    # Main content area
    if st.session_state.df is None:
        st.title("💬 WhatsApp Chat Analyzer")
        st.warning("No chat data loaded. Please upload a WhatsApp chat export and index it to begin.")
        return

    if not st.session_state.is_indexed:
        st.info("Chat uploaded. Please use the sidebar to index your chat data to unlock all analysis features.")
    
    # Route to the selected page
    current_page = st.session_state.current_page
    
    if current_page == "Home" or current_page == "📊 Analytics":
        st.header("📊 Chat Analytics")
        try:
            from ui.page_modules import analytics_page
            analytics_page.show(st.session_state.df)
        except Exception as e:
            st.error(f"❌ Error loading analytics: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    
    elif current_page == "❓ Q&A":
        st.header("❓ Ask About Your Chat")
        try:
            if not st.session_state.is_indexed:
                st.warning("⚠️ Please index your chat data first using the sidebar button")
            else:
                from ui.page_modules import rag_qa_page
                rag_qa_page.show(st.session_state.df)
        except Exception as e:
            st.error(f"❌ Error loading Q&A: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    
    elif current_page == "💭 Sentiment":
        st.header("💭 Sentiment Analysis")
        try:
            if not st.session_state.is_indexed:
                st.warning("⚠️ Please index your chat data first using the sidebar button")
            else:
                from ui.page_modules import sentiment_page
                sentiment_page.show(st.session_state.df)
        except Exception as e:
            st.error(f"❌ Error loading sentiment analysis: {str(e)}")
            import traceback
            st.debug(traceback.format_exc())
    
    elif current_page == "📚 Topics":
        st.header("📚 Topic Extraction")
        try:
            if not st.session_state.is_indexed:
                st.warning("⚠️ Please index your chat data first using the sidebar button")
            else:
                from ui.page_modules import topics_page
                topics_page.show(st.session_state.df)
        except Exception as e:
            st.error(f"❌ Error loading topic modeling: {str(e)}")
            import traceback
            st.debug(traceback.format_exc())

if __name__ == "__main__":
    main()