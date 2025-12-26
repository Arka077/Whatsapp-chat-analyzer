"""
Sentiment Analysis Page - Fixed Version
"""

import streamlit as st
from analytics.sentiment_analyzer import TimeBasedSentimentAnalyzer
from utils.date_utils import get_preset_ranges, validate_date_range
import pandas as pd
import plotly.graph_objects as go

def show(df):
    """Display sentiment analysis page"""
    
    st.markdown("""
    Analyze the sentiment of your conversations over time. 
    Select a date range and the system will analyze emotions and sentiment patterns.
    """)
    
    st.divider()
    
    # --- Analysis Options in main page ---
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
    
    # --- Analysis Execution ---
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
                st.exception(traceback.format_exc())
    
    # --- Display Results ---
    if 'sentiment_result' in st.session_state:
        result = st.session_state.sentiment_result
        
        st.markdown("---")
        st.header("📊 Sentiment Analysis Results")
        
        if 'error' not in result:
            # 1. Overall Metrics
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
            
            # 2. Charts Section (Pie + Bar)
            col1, col2 = st.columns(2)
            
            # --- LEFT COLUMN: Sentiment Pie Chart ---
            with col1:
                st.markdown("**Sentiment Distribution**")
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Neutral', 'Negative'],
                    values=[
                        sentiment.get('positive', 0),
                        sentiment.get('neutral', 0),
                        sentiment.get('negative', 0)
                    ],
                    marker=dict(colors=['#2ecc71', '#95a5a6', '#e74c3c']),
                    hole=0.4, # Donut style
                    textinfo='percent+label',
                    textposition='inside'
                )])
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                
                # FIXED: use_container_width=True
                st.plotly_chart(fig, use_container_width=True)
            
            # --- RIGHT COLUMN: Emotions Bar Chart ---
            with col2:
                st.markdown("**Emotions Detected**")
                emotions = result.get('emotions', {})
                
                # Filter out zero values so graph isn't empty
                emotions = {k: v for k, v in emotions.items() if v > 0}
                
                if emotions:
                    emotion_labels = list(emotions.keys())
                    emotion_values = list(emotions.values())
                    
                    # Define colors
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
                        marker=dict(color=colors),
                        text=emotion_values,
                        texttemplate='%{text:.1%}',
                        textposition='auto'
                    )])
                    
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        yaxis=dict(
                            title="Score",
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.1)'
                        ),
                        xaxis=dict(showgrid=False),
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    
                    # FIXED: use_container_width=True
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No specific emotions detected")
            
            st.divider()
            
            # 3. Daily Breakdown Chart
            st.subheader("📈 Daily Breakdown")
            
            # PATCH: Calculate daily counts directly from dataframe instead of relying on 'result'
            # This ensures the graph works even if the analyzer backend fails to group dates
            try:
                # Ensure date column is string for comparison
                mask = (df['only_date'].astype(str) >= str(start_date)) & (df['only_date'].astype(str) <= str(end_date))
                filtered_df = df.loc[mask]
                
                # Group by date and count messages
                daily_df = filtered_df.groupby('only_date').size().reset_index(name='message_count')
                daily_df = daily_df.rename(columns={'only_date': 'date'}) # Rename to match plotting code
                
                if not daily_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=daily_df['date'],
                        y=daily_df['message_count'],
                        name='Messages',
                        marker_color='#3498db', # Bright blue
                        text=daily_df['message_count'],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        height=400, 
                        hovermode='x unified',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        xaxis=dict(
                            showgrid=False,
                            type='category' # Ensures dates aren't skipped
                        ),
                        yaxis=dict(
                            showgrid=True, 
                            gridcolor='rgba(255,255,255,0.1)',
                            title="Message Count"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No daily data available for this selection.")
                    
            except Exception as e:
                st.error(f"Error generating daily chart: {e}")
            
            st.divider()
            
            # 4. Insights & Export
            st.subheader("💡 Insights")
            st.info(result.get('insights', 'No insights available'))
            
            st.divider()
            
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
