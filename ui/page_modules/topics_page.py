"""
Topic Modeling Page
"""

import streamlit as st
from analytics.topic_modeler import TopicModeler
from utils.date_utils import get_preset_ranges, validate_date_range
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def show(df):
    """Display topic modeling page"""
    
    st.markdown("""
    Extract and analyze topics from your conversations.
    The system will identify main discussion themes and analyze their sentiment.
    """)
    
    st.divider()
    
    # Topic Settings in main page
    st.subheader("⚙️ Topic Settings")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_topics = st.slider(
            "Number of topics to extract",
            min_value=2,
            max_value=10,
            value=5,
            step=1
        )
    
    with col2:
        # Date range selection
        st.markdown("**📅 Date Range**")
        
        date_preset = st.selectbox(
            "Select period",
            list(get_preset_ranges().keys()) + ["Custom"],
            key="topics_date_preset"
        )
        
        if date_preset == "Custom":
            col_s, col_e = st.columns(2)
            with col_s:
                start_date_obj = st.date_input(
                    "Start date",
                    value=df['only_date'].min(),
                    min_value=df['only_date'].min(),
                    max_value=df['only_date'].max()
                )
                start_date = start_date_obj.strftime("%Y-%m-%d")
            with col_e:
                end_date_obj = st.date_input(
                    "End date",
                    value=df['only_date'].max(),
                    min_value=df['only_date'].min(),
                    max_value=df['only_date'].max()
                )
                end_date = end_date_obj.strftime("%Y-%m-%d")
        else:
            start_date, end_date = get_preset_ranges()[date_preset]
            # Convert strings to date objects for comparison
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if not validate_date_range(start_date, end_date):
            st.error("Invalid date range")
            return
    
    # Extract topics button
    if st.button(f"🔍 Extract Topics", type="primary"):
        with st.spinner(f"⏳ Extracting {num_topics} topics..."):
            try:
                # Check if we have enough messages
                # Use date objects for comparison with df['only_date']
                messages_in_range = df[
                    (df['only_date'] >= start_date_obj) & 
                    (df['only_date'] <= end_date_obj) &
                    (df['user'] != 'group_notification') &
                    (df['message'] != '<Media omitted>')
                ]
                
                if len(messages_in_range) < 10:
                    st.warning(f"⚠️ Only {len(messages_in_range)} messages found in date range. Need at least 10 for topic modeling.")
                    return
                
                modeler = TopicModeler()
                result = modeler.extract_topics(df, start_date, end_date, num_topics)
                st.session_state.topics_result = result
            
            except Exception as e:
                st.error(f"❌ Error extracting topics: {str(e)[:150]}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    
    # Display results
    if 'topics_result' in st.session_state:
        result = st.session_state.topics_result
        
        st.markdown("---")
        st.header("📚 Extracted Topics")
        
        if 'error' not in result and result.get('topics'):
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Topics Found", len(result['topics']))
            with col2:
                st.metric("Date Range", f"{start_date} to {end_date}")
            with col3:
                st.metric("Messages Analyzed", result.get('total_messages', 0))
            
            st.divider()
            
            # Topics list
            st.subheader("🎯 Topics")
            
            for i, topic in enumerate(result['topics'], 1):
                # Handle both 'topic_name' and 'name' keys for compatibility
                topic_name = topic.get('topic_name') or topic.get('name', 'Unknown')
                
                with st.expander(f"Topic {i}: {topic_name}", expanded=(i == 1)):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Description:**")
                        st.write(topic.get('description', 'N/A'))
                        
                        # Handle both 'key_keywords' and 'keywords' keys
                        keywords = topic.get('key_keywords') or topic.get('keywords', [])
                        if keywords:
                            st.markdown("**Keywords:**")
                            st.write(", ".join(f"`{k}`" for k in keywords))
                        
                        # Show participants if available
                        participants = topic.get('participants', [])
                        if participants and isinstance(participants, list) and len(participants) > 0:
                            st.markdown("**Participants:**")
                            st.write(", ".join(participants))
                        
                        # Show message count if available
                        msg_count = topic.get('message_count', 0)
                        if msg_count > 0:
                            st.markdown(f"**Messages in Topic:** {msg_count}")
                    
                    with col2:
                        st.markdown("**Topic Sentiment:**")
                        sentiment = topic.get('sentiment', {})
                        st.metric("Positive", f"{sentiment.get('positive', 0):.0%}")
                        st.metric("Neutral", f"{sentiment.get('neutral', 0):.0%}")
                        st.metric("Negative", f"{sentiment.get('negative', 0):.0%}")
            
            st.divider()
            
            # Insights
            st.subheader("💡 Analysis Summary")
            summary = result.get('analysis_summary') or result.get('insights', 'No insights available')
            st.info(summary)
            
            st.divider()
            
            # Export
            st.subheader("📥 Export")
            
            if st.button("📋 Export Topics as JSON"):
                import json
                export_data = {
                    'date_range': result.get('date_range'),
                    'total_messages': result.get('total_messages'),
                    'topics': result.get('topics'),
                    'analysis_summary': result.get('analysis_summary') or result.get('insights')
                }
                
                st.json(export_data)
        
        else:
            error_msg = result.get('insights') or result.get('analysis_summary') or result.get('error', 'Could not extract topics')
            st.warning(error_msg)