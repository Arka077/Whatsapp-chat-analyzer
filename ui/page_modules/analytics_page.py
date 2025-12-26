"""
Analytics Dashboard Page
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from urlextract import URLExtract
import emoji
import os

extract = URLExtract()

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Color Emoji']

def load_stopwords():
    """Load stopwords from data folder"""
    stopwords = set()
    stopwords_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'stopwords')
    
    # Load all stopword files
    stopword_files = ['stop_benglish.txt', 'stop_hinglish.txt', 'stop_words_english.txt']
    
    for filename in stopword_files:
        filepath = os.path.join(stopwords_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    words = f.read().split()
                    stopwords.update(words)
            except:
                pass
    
    return stopwords

def show(df):
    """Display analytics dashboard"""
    
    try:
        # User selection
        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")
        
        selected_user = st.selectbox("Select user", user_list, key="analytics_user_select")
        
        # Filter data
        if selected_user != "Overall":
            display_df = df[df['user'] == selected_user]
        else:
            display_df = df
        
        # Top Statistics
        st.subheader("📈 Key Metrics")
        
        # Calculate stats
        filtered_df = display_df[
            (display_df['user'] != 'group_notification') & 
            (display_df['message'].str.strip() != '')
        ]
        
        # Handle missing is_system_message column safely
        if 'is_system_message' in filtered_df.columns:
            filtered_df = filtered_df[~filtered_df['is_system_message']]
        
        num_messages = len(filtered_df)
        words = sum(len(str(msg).split()) for msg in filtered_df['message'] if str(msg).strip())
        media_messages = len(display_df[display_df['message'] == '<Media omitted>'])
        
        try:
            links = sum(len(extract.find_urls(str(msg))) for msg in display_df['message'] if msg)
        except:
            links = 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Messages", num_messages)
        with col2:
            st.metric("Words", words)
        with col3:
            st.metric("Media", media_messages)
        with col4:
            st.metric("Links", links)
        
        st.divider()
        
        # Monthly Timeline
        st.subheader("📅 Activity Over Time")
        try:
            timeline = display_df.groupby(['year', 'month_num', 'month']).size().reset_index(name='message')
            
            if len(timeline) > 0:
                time_labels = [f"{row['month']}-{row['year']}" for _, row in timeline.iterrows()]
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(range(len(time_labels)), timeline['message'].values, marker='o', color='#1f77b4', linewidth=2)
                ax.fill_between(range(len(time_labels)), timeline['message'].values, alpha=0.3, color='#1f77b4')
                ax.set_xticks(range(len(time_labels)))
                ax.set_xticklabels(time_labels, rotation=45)
                ax.set_ylabel("Messages")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, width='stretch')
        except Exception as e:
            st.warning(f"Could not display monthly timeline: {str(e)[:50]}")
        
        st.divider()
        
        # Activity Patterns
        st.subheader("⏰ Activity Patterns")
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Busiest Days**")
                busy_day = display_df['day_name'].value_counts()
                if len(busy_day) > 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(busy_day.index, busy_day.values, color='#2ca02c', alpha=0.7)
                    plt.xticks(rotation=45)
                    ax.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig, width='stretch')
            
            with col2:
                st.markdown("**Busiest Months**")
                busy_month = display_df['month'].value_counts()
                if len(busy_month) > 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(busy_month.index, busy_month.values, color='#ff7f0e', alpha=0.7)
                    plt.xticks(rotation=45)
                    ax.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig, width='stretch')
        except Exception as e:
            st.warning(f"Could not display activity patterns: {str(e)[:50]}")
        
        st.divider()
        
        # Busiest Users (for Overall)
        if selected_user == "Overall":
            st.subheader("👥 Most Active Users")
            try:
                busy_users = df[df['user'] != 'group_notification']['user'].value_counts().head(10)
                
                if len(busy_users) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.barh(busy_users.index, busy_users.values, color='#d62728', alpha=0.7)
                        ax.grid(True, alpha=0.3, axis='x')
                        st.pyplot(fig, width='stretch')
                    
                    with col2:
                        user_stats = pd.DataFrame({
                            'User': busy_users.index,
                            'Messages': busy_users.values,
                            '%': (busy_users.values / busy_users.sum() * 100).round(2)
                        })
                        st.dataframe(user_stats, width='stretch', hide_index=True)
                
                st.divider()
            except Exception as e:
                st.warning(f"Could not display user stats: {str(e)[:50]}")
        
        # Word Cloud
        st.subheader("☁️ Word Cloud")
        
        try:
            # Filter messages for wordcloud - exclude system messages
            wc_df = display_df[
                (display_df['user'] != 'group_notification') &
                (display_df['message'] != '<Media omitted>')
            ]
            
            if len(wc_df) > 0:
                # Load stopwords
                stopwords = load_stopwords()
                
                # Clean text: remove URLs, media markers, and extra whitespace
                import re
                text = " ".join(wc_df['message'].astype(str))
                text = re.sub(r'<media\s+omitted>', '', text, flags=re.IGNORECASE)
                text = re.sub(r'http\S+|www\S+', '', text)
                text = re.sub(r'\s+', ' ', text)
                
                if text.strip():
                    wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig, width='stretch')
                else:
                    st.info("Not enough text to generate word cloud")
        except Exception as e:
            st.warning(f"Could not generate wordcloud: {str(e)[:50]}")
        
        st.divider()
        
        # Most Common Words
        st.subheader("🔤 Most Common Words")
        
        try:
            # Load stopwords
            stopwords = load_stopwords()
            
            words_list = []
            for idx, msg in display_df[display_df['user'] != 'group_notification']['message'].items():
                # Remove URLs and media markers from message
                msg_clean = str(msg).lower()
                msg_clean = re.sub(r'<media\s+omitted>', '', msg_clean, flags=re.IGNORECASE)
                msg_clean = re.sub(r'http\S+|www\S+', '', msg_clean)
                words_list.extend(msg_clean.split())
            
            # Remove stopwords and short words
            words_list = [w for w in words_list if w not in stopwords and len(w) > 2 and not w.startswith('http') and not w.startswith('<')]
            
            most_common = Counter(words_list).most_common(20)
            
            if most_common:
                words_df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(words_df['Word'], words_df['Frequency'], color='#1f77b4', alpha=0.7)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig, width='stretch')
        except Exception as e:
            st.warning(f"Could not analyze words: {str(e)[:50]}")
    
    except Exception as e:
        st.error(f"❌ Error loading analytics: {str(e)[:150]}")
        import traceback
        st.debug(traceback.format_exc())
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(busy_users.index, busy_users.values, color='red', alpha=0.7)
            st.pyplot(fig)
        
        with col2:
            user_stats = pd.DataFrame({
                'User': busy_users.index,
                'Messages': busy_users.values,
                'Percentage': (busy_users.values / busy_users.sum() * 100).round(2)
            })
            st.dataframe(user_stats, width='stretch')
        
        st.divider()
    
    
    # Response Time Analysis
    st.subheader("⏱️ Response Time Analysis")
    
    try:
        # Calculate response times for each user
        response_times = []
        
        # Sort by date to get sequential order
        sorted_df = display_df.sort_values('date').copy()
        
        for i in range(1, len(sorted_df)):
            current_msg = sorted_df.iloc[i]
            prev_msg = sorted_df.iloc[i-1]
            
            # Only calculate if different users
            if current_msg['user'] != prev_msg['user'] and prev_msg['user'] != 'group_notification':
                time_diff = current_msg['date'] - prev_msg['date']
                # Convert to minutes
                minutes_diff = time_diff.total_seconds() / 60
                
                response_times.append({
                    'user': current_msg['user'],
                    'previous_user': prev_msg['user'],
                    'response_time_minutes': minutes_diff
                })
        
        if response_times:
            response_df = pd.DataFrame(response_times)
            
            # Calculate average response time per user
            avg_response = response_df.groupby('user')['response_time_minutes'].agg(['mean', 'median', 'count']).reset_index()
            avg_response.columns = ['User', 'Avg Response (min)', 'Median Response (min)', 'Responses']
            avg_response = avg_response.sort_values('Avg Response (min)')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Average Response Time by User**")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(avg_response['User'], avg_response['Avg Response (min)'], color='#ff7f0e', alpha=0.7)
                ax.set_xlabel('Average Response Time (minutes)')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig, width='stretch')
            
            with col2:
                st.markdown("**Response Time Statistics**")
                st.dataframe(avg_response, width='stretch', hide_index=True)
        else:
            st.info("Not enough messages from different users to calculate response times")
    
    except Exception as e:
        st.warning(f"Could not analyze response times: {str(e)[:50]}")
    
    st.divider()
    
    # Talkativeness & Messaging Trends
    st.subheader("💬 Talkativeness & Messaging Trends")
    
    try:
        # Get all users (excluding group notifications)
        all_users = df[df['user'] != 'group_notification']['user'].unique().tolist()
        all_users.sort()
        
        if selected_user != "Overall":
            # For individual user, show their daily messaging pattern
            user_daily = display_df.groupby('only_date').size().reset_index(name='messages')
            
            if len(user_daily) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Daily Message Count**")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.bar(range(len(user_daily)), user_daily['messages'].values, color='#2ca02c', alpha=0.7)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Messages')
                    ax.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig, width='stretch')
                
                with col2:
                    st.markdown("**Messaging Stats**")
                    stats_data = {
                        'Metric': ['Total Messages', 'Daily Average', 'Peak Day', 'Quiet Day'],
                        'Value': [
                            len(display_df),
                            f"{display_df.groupby('only_date').size().mean():.1f}",
                            f"{display_df.groupby('only_date').size().max()}",
                            f"{display_df.groupby('only_date').size().min()}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), width='stretch', hide_index=True)
        
        else:
            # For overall, show talkativeness comparison
            user_stats = []
            for user in all_users:
                user_df = df[df['user'] == user]
                user_stats.append({
                    'User': user,
                    'Total Messages': len(user_df),
                    'Avg Daily': len(user_df) / max(1, (user_df['only_date'].max() - user_df['only_date'].min()).days + 1),
                    'Total Words': sum(len(str(msg).split()) for msg in user_df['message'] if str(msg).strip())
                })
            
            talkativeness_df = pd.DataFrame(user_stats).sort_values('Total Messages', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**User Talkativeness Ranking**")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(talkativeness_df['User'], talkativeness_df['Total Messages'], color='#d62728', alpha=0.7)
                ax.set_xlabel('Total Messages')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig, width='stretch')
            
            with col2:
                st.markdown("**Detailed User Stats**")
                display_stats = talkativeness_df[['User', 'Total Messages', 'Avg Daily', 'Total Words']].copy()
                display_stats['Avg Daily'] = display_stats['Avg Daily'].round(2)
                display_stats['Total Words'] = display_stats['Total Words'].astype(int)
                st.dataframe(display_stats, width='stretch', hide_index=True)
            
            # Messaging trend over time by top users
            st.markdown("**Messaging Trends - Top 3 Users**")
            
            top_users = talkativeness_df.head(3)['User'].tolist()
            
            try:
                fig, ax = plt.subplots(figsize=(12, 5))
                
                for user in top_users:
                    user_df = df[df['user'] == user]
                    daily_count = user_df.groupby('only_date').size()
                    ax.plot(daily_count.index, daily_count.values, marker='o', label=user, linewidth=2, alpha=0.7)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Messages per Day')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig, width='stretch')
            except Exception as e:
                st.warning(f"Could not display trend chart: {str(e)[:50]}")
    
    except Exception as e:
        st.warning(f"Could not analyze talkativeness: {str(e)[:50]}")
    
    st.divider()
    
    # Emoji Analysis
    st.subheader("😊 Emoji Analysis")
    
    try:
        emojis = []
        emoji_list = emoji.emoji_list if hasattr(emoji, 'emoji_list') else []
        
        for msg in display_df['message']:
            for char in str(msg):
                try:
                    if emoji.is_emoji(char):
                        emojis.append(char)
                except:
                    pass
        
        if emojis:
            emoji_counter = Counter(emojis).most_common(10)
            emoji_df = pd.DataFrame(emoji_counter, columns=['Emoji', 'Count'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df, width='stretch')
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.pie(emoji_df['Count'], labels=emoji_df['Emoji'], autopct='%1.1f%%')
                st.pyplot(fig)
        else:
            st.info("No emojis found in chat")
    
    except Exception as e:
        st.warning(f"Could not analyze emojis: {e}")
