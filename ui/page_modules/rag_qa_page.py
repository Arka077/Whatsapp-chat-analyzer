"""
RAG Q&A Page - Professional Chat Analysis Interface
"""

import streamlit as st
from rag. rag_pipeline import RAGPipeline
from utils.date_utils import get_preset_ranges, validate_date_range
import pandas as pd
from datetime import datetime

def show(df):
    """Display RAG Q&A interface"""
    
    # Initialize session state for Q&A history
    if 'qa_history' not in st. session_state:
        st. session_state.qa_history = []
    if 'show_citations' not in st.session_state:
        st.session_state. show_citations = False
    
    # Inline Configuration Panel
    st.markdown("### 🔧 Configuration")
    config_container = st.container()
    with config_container:
        config_col_left, config_col_right = st.columns([3, 2], gap="large")
        
        with config_col_left:
            st. markdown("**📅 Date Range**")
            date_preset = st.selectbox(
                "Select date range",
                ["All Time"] + list(get_preset_ranges().keys()) + ["Custom"],
                key="rag_date_preset",
                label_visibility="collapsed"
            )
            
            start_date = None
            end_date = None
            
            if date_preset == "All Time":
                start_date = df['only_date'].min().strftime("%Y-%m-%d")
                end_date = df['only_date'].max().strftime("%Y-%m-%d")
            elif date_preset == "Custom":
                custom_col1, custom_col2 = st.columns(2)
                with custom_col1:
                    start_date = st.date_input(
                        "Start",
                        value=df['only_date'].min(),
                        min_value=df['only_date'].min(),
                        max_value=df['only_date'].max(),
                        label_visibility="collapsed",
                        key="rag_custom_start"
                    ).strftime("%Y-%m-%d")
                with custom_col2:
                    end_date = st.date_input(
                        "End",
                        value=df['only_date'].max(),
                        min_value=df['only_date'].min(),
                        max_value=df['only_date'].max(),
                        label_visibility="collapsed",
                        key="rag_custom_end"
                    ).strftime("%Y-%m-%d")
            else:
                start_date, end_date = get_preset_ranges()[date_preset]
        
        with config_col_right:
            st.markdown("**⚙️ Retrieval Settings**")
            top_k = st.slider(
                "Messages to retrieve",
                min_value=5,
                max_value=100,
                value=50,
                step=5,
                label_visibility="collapsed",
                help="Higher values provide more context",
                key="rag_top_k_slider"
            )
            
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.caption(f"📍 **{start_date}**")
        with meta_col2:
            st.caption(f"**{end_date}**")
    
    # Validate date range
    if not validate_date_range(start_date, end_date):
        st.error("Invalid date range")
        return
    
    # Main Content Area
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <p style="color: #666; font-size:  0.95rem;">
        Ask natural questions about your conversations and get instant answers with optional citations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Question Input Section
    st.markdown("**Your Question**")
    col1, col2 = st. columns([5, 1], gap="small")
    
    with col1:
        question = st.text_input(
            "Ask about your chat",
            placeholder="e.g., What is Didi talking about most?  When is the deadline?",
            label_visibility="collapsed",
            key="qa_input"
        )
    
    with col2:
        ask_button = st.button("🔍 Ask", use_container_width=True, key="ask_btn")
    
    # Process Question
    # Ensure we have a pipeline tied to the currently selected index
    active_index = st.session_state.get('index_name')
    if (
        'rag_pipeline' not in st.session_state
        or st.session_state. get('rag_pipeline_index') != active_index
    ):
        st.session_state.rag_pipeline = RAGPipeline(index_name=active_index)
        st.session_state.rag_pipeline_index = active_index
    
    pipeline = st.session_state.get('rag_pipeline')
    
    if ask_button and question.strip():
        with st.spinner("🔍 Searching conversations..."):
            try:
                result = pipeline.answer_question(
                    query=question,
                    date_range=(start_date, end_date),
                    top_k=top_k
                )
                
                # Store in history
                st.session_state. qa_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'citations': result['citations'],
                    'confidence': result['confidence'],
                    'sources_count': result['sources_count'],
                    'timestamp': datetime. now()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                if "'" in str(e) and "message" in str(e):
                    st.info("💡 **Tip:** Make sure your chat data is indexed properly.  Try re-uploading and indexing.")
    
    # Display Latest Result
    if st.session_state.qa_history:
        latest = st.session_state.qa_history[-1]
        
        st.markdown("---")
        
        # Answer Container with Better Styling
        st.markdown("<h3 style='margin-bottom: 0.5rem;'>✅ Answer</h3>", unsafe_allow_html=True)
        
        # Professional answer box
        answer_box = f"""
        <div style="
            background-color: #f0f4f8;
            padding: 1.5rem;
            border-left: 4px solid #1f77b4;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
        ">
            <p style="margin:  0; font-size: 1rem; line-height: 1.6; color: #1a1a1a;">
            {latest['answer']}
            </p>
        </div>
        """
        st.markdown(answer_box, unsafe_allow_html=True)
        
        # Metrics Row
        col1, col2, col3 = st.columns(3, gap="small")
        with col1:
            st.metric("Confidence", f"{latest['confidence']:.0%}", delta=None)
        with col2:
            st.metric("Sources", latest['sources_count'], delta=None)
        with col3:
            st.metric("Retrieved", f"{top_k} messages", delta=None)
        
        # Optional Citations Section
        if latest.get('citations'):
            st.markdown("---")
            
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown("<h4>📌 Original Messages</h4>", unsafe_allow_html=True)
            with col2:
                show_citations_toggle = st.checkbox(
                    "Show",
                    value=False,
                    label_visibility="collapsed",
                    key="citations_toggle"
                )
            
            if show_citations_toggle:
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>Relevant messages from your chat: </p>", unsafe_allow_html=True)
                
                for i, citation in enumerate(latest['citations'], 1):
                    # Handle both old and new citation formats
                    user = citation.get('user', 'Unknown')
                    timestamp = citation.get('timestamp', citation. get('date', 'N/A'))
                    message = citation.get('message', citation.get('text', 'No message content'))
                    score = citation. get('score', citation.get('similarity_score', 0.0))
                    
                    with st.expander(
                        f"📨 {user} • {timestamp}",
                        expanded=(i <= 2)
                    ):
                        st. write(message)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"👤 {user}")
                        with col2:
                            st.caption(f"⏰ {timestamp}")
                        with col3:
                            st.caption(f"🎯 {score:.0%} match")
    
    # Question History Section
    if st.session_state.qa_history:
        st.markdown("---")
        st.markdown("<h3>📜 Recent Questions</h3>", unsafe_allow_html=True)
        
        for i, item in enumerate(reversed(st.session_state.qa_history), 1):
            q_num = len(st.session_state.qa_history) - i + 1
            
            with st.expander(
                f"**Q{q_num}:** {item['question'][:70]}{'...' if len(item['question']) > 70 else ''}",
                expanded=False
            ):
                st.write(f"**Question:** {item['question']}")
                st. write(f"**Answer:** {item['answer']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"📊 Confidence: {item['confidence']:.0%}")
                with col2:
                    st. caption(f"📌 Sources: {item['sources_count']}")
    else:
        if not (ask_button and question. strip()):
            st.info("💬 Ask a question above to get started!")
