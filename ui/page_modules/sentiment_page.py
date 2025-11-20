"""
Sentiment Analysis Page
"""

import streamlit as st
from analytics.sentiment_analyzer import TimeBasedSentimentAnalyzer
from utils.date_utils import get_preset_ranges, validate_date_range
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def show(df):
    """Display sentiment analysis page"""
    
    st.markdown("""
    Analyze the sentiment of your conversations over time. 
    Select a date range and the system will analyze emotions and sentiment patterns.
    """)
    
    st.divider()
    
    # Analysis Options in main page
    st.subheader("⚙️ Analysis Options")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        analysis_type = st.radio(
            "Analysis Type",
            ["Overall", "By User", "By Topic"],
            key="sentiment_analysis_type"
        )
    
    with col2:
        # Date range selection
        st.markdown("**📅 Date Range**")
        
        date_preset = st.selectbox(
            "Select period",
            list(get_preset_ranges().keys()) + ["Custom"],
            key="sentiment_date_preset"
        )
        
        if date_preset == "Custom":
            col_s, col_e = st.columns(2)
            with col_s:
                start_date = st.date_input(
                    "Start date",
                    value=df['only_date'].min(),
                    min_value=df['only_date'].min(),
                    max_value=df['only_date'].max()
                ).strftime("%Y-%m-%d")
            with col_e:
                end_date = st.date_input(
                    "End date",
                    value=df['only_date'].max(),
                    min_value=df['only_date'].min(),
                    max_value=df['only_date'].max()
                ).strftime("%Y-%m-%d")
        else:
            start_date, end_date = get_preset_ranges()[date_preset]
        
        if not validate_date_range(start_date, end_date):
            st.error("Invalid date range")
            return
        
        # User selection for "By User" analysis
        selected_user = None
        if analysis_type == "By User":
            user_list = df['user'].unique().tolist()
            if 'group_notification' in user_list:
                user_list.remove('group_notification')
            user_list.sort()
            
            if user_list:
                selected_user = st.selectbox("Select user", user_list, key="sentiment_user_select")
            else:
                st.warning("No users found")
                return
    
    # Analysis execution button
    if st.button("📊 Analyze Sentiment", type="primary"):
        with st.spinner("⏳ Analyzing sentiment..."):
            try:
                analyzer = TimeBasedSentimentAnalyzer()
                result = None
                
                if analysis_type == "Overall":
                    result = analyzer.analyze_date_range(df, start_date, end_date)
                
                elif analysis_type == "By User":
                    if selected_user:
                        result = analyzer.analyze_by_user(df, selected_user, start_date, end_date)
                        if result:
                            result['analysis_type'] = f"User: {selected_user}"
                
                else:  # By Topic
                    st.info("💡 Topic-based sentiment will be shown after topic modeling")
                    result = None
                
                if result:
                    st.session_state.sentiment_result = result
            
            except Exception as e:
                st.error(f"❌ Error analyzing sentiment: {str(e)[:150]}")
                import traceback
                st.debug(traceback.format_exc())
    
    # Display results
    if 'sentiment_result' in st.session_state:
        result = st.session_state.sentiment_result
        
        st.markdown("---")
        st.header("📊 Sentiment Analysis Results")
        
        if 'error' not in result:
            # Overall sentiment
            st.subheader("Overall Sentiment")
            
            col1, col2, col3, col4 = st.columns(4)
            
            sentiment = result.get('overall_sentiment', {})
            with col1:
                st.metric("Positive", f"{sentiment.get('positive', 0):.1%}")
            with col2:
                st.metric("Neutral", f"{sentiment.get('neutral', 0):.1%}")
            with col3:
                st.metric("Negative", f"{sentiment.get('negative', 0):.1%}")
            with col4:
                st.metric("Total Messages", result.get('total_messages', 0))
            
            # Sentiment pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sentiment Distribution**")
                fig = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Neutral', 'Negative'],
                    values=[
                        sentiment.get('positive', 0),
                        sentiment.get('neutral', 0),
                        sentiment.get('negative', 0)
                    ],
                    marker=dict(colors=['#2ecc71', '#95a5a6', '#e74c3c'])
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Emotions
            with col2:
                st.markdown("**Emotions Detected**")
                emotions = result.get('emotions', {})
                if emotions:
                    emotion_labels = list(emotions.keys())
                    emotion_values = list(emotions.values())
                    
                    # Define colors for emotions (positive=green, negative=red, neutral=gray)
                    positive_emotions = {'joy', 'enthusiasm', 'excitement'}
                    negative_emotions = {'anger', 'frustration', 'sadness', 'concern'}
                    
                    colors = []
                    for emotion in emotion_labels:
                        if emotion in positive_emotions:
                            colors.append('#2ecc71')  # Green
                        elif emotion in negative_emotions:
                            colors.append('#e74c3c')  # Red
                        else:
                            colors.append('#95a5a6')  # Gray
                    
                    fig = go.Figure(data=[go.Bar(
                        x=emotion_labels,
                        y=emotion_values,
                        marker=dict(color=colors)
                    )])
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        yaxis_title="Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No specific emotions detected")
            
            st.divider()
            
            # Daily breakdown
            st.subheader("📈 Daily Breakdown")
            
            daily = result.get('daily_breakdown', [])
            if daily:
                daily_df = pd.DataFrame(daily)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=daily_df['date'],
                    y=daily_df['message_count'],
                    name='Messages',
                    marker_color='rgb(55, 83, 109)'
                ))
                fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Insights
            st.subheader("💡 Insights")
            st.info(result.get('insights', 'No insights available'))
            
            st.divider()
            
            # Export options
            st.subheader("📥 Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy Analysis as Text"):
                    text_export = f"""
Sentiment Analysis Report
Date Range: {start_date} to {end_date}
Total Messages: {result.get('total_messages', 0)}

Overall Sentiment:
- Positive: {sentiment.get('positive', 0):.1%}
- Neutral: {sentiment.get('neutral', 0):.1%}
- Negative: {sentiment.get('negative', 0):.1%}

Emotions:
{chr(10).join([f"- {k}: {v:.1%}" for k, v in emotions.items()])}

Insights:
{result.get('insights', 'N/A')}
"""
                    st.text_area("Export as Text", value=text_export, height=200)
        else:
            st.error(result.get('insights', 'Error analyzing sentiment'))