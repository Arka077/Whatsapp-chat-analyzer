"""
Time-based Sentiment Analysis using Gemini
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime
from llm.gemini_client import GeminiClient
from llm.prompt_templates import get_sentiment_prompt, get_sentiment_scores_prompt, get_topic_sentiment_prompt
import json

class TimeBasedSentimentAnalyzer:
    """Analyze sentiment of messages in date range"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.llm = GeminiClient()
    
    def analyze_date_range(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Analyze sentiment for messages in date range
        
        Args:
            df: preprocessed chat dataframe
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
        
        Returns:
            Dict with sentiment analysis results
        """
        try:
            # Filter messages by date range
            df_filtered = df[
                (df['only_date'].astype(str) >= start_date) &
                (df['only_date'].astype(str) <= end_date) &
                (df['user'] != 'group_notification') &
                (df['message'] != '<Media omitted>') &
                (df['message'].str.strip() != '')
            ].copy()
            
            if len(df_filtered) == 0:
                return {
                    "date_range": {"start": start_date, "end": end_date},
                    "overall_sentiment": {"positive": 0, "neutral": 1, "negative": 0},
                    "emotions": {},
                    "daily_breakdown": [],
                    "insights": "No messages found in this date range.",
                    "total_messages": 0
                }
            
            # First, get sentiment scores from Gemini
            date_range_str = f"{start_date} to {end_date}"
            scores_prompt = get_sentiment_scores_prompt(df_filtered.to_dict('records'), date_range_str)
            scores_response = self.llm.generate_json(scores_prompt)
            
            # Parse sentiment scores
            if scores_response and 'positive_score' in scores_response:
                overall_sentiment = {
                    "positive": float(scores_response.get('positive_score', 0.0)),
                    "neutral": float(scores_response.get('neutral_score', 0.0)),
                    "negative": float(scores_response.get('negative_score', 0.0))
                }
                
                # Normalize to ensure they sum to 1.0
                total = sum(overall_sentiment.values())
                if total > 0:
                    overall_sentiment = {k: v/total for k, v in overall_sentiment.items()}
            else:
                # Fallback if scores prompt fails
                overall_sentiment = self._calculate_sentiment_scores(df_filtered)
            
            # Then get detailed emotions analysis
            prompt = get_sentiment_prompt(df_filtered.to_dict('records'), date_range_str)
            response = self.llm.generate_json(prompt)
            
            # Validate and use response if valid
            if response and isinstance(response, dict) and 'emotions' in response:
                emotions = response.get('emotions', {})
                insights = response.get('insights', "Analysis completed using Gemini AI")
            else:
                # Use basic emotion calculation
                emotions = self._calculate_emotions(df_filtered)
                insights = "Sentiment analysis completed"
            
            return {
                "date_range": {"start": start_date, "end": end_date},
                "overall_sentiment": overall_sentiment,
                "emotions": emotions,
                "daily_breakdown": self._get_daily_breakdown(df_filtered),
                "insights": insights,
                "total_messages": len(df_filtered),
                "primary_emotion": max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
            }
        
        except Exception as e:
            print(f"[DEBUG] Sentiment analysis error: {str(e)}")
            return self._basic_sentiment_analysis(df, start_date, end_date)
    
    def _calculate_sentiment_scores(self, df: pd.DataFrame) -> Dict:
        """Calculate sentiment scores using keyword matching"""
        positive_words = {'good', 'great', 'awesome', 'amazing', 'happy', 'love', 'excellent', 'wonderful', 'fantastic', 'nice', 'best', 'perfect', 'cool', 'party', 'yes', 'yay', 'lol', ':)', '😊', '❤️', 'thank', 'thanks'}
        negative_words = {'bad', 'hate', 'angry', 'sad', 'terrible', 'awful', 'horrible', 'worst', 'stupid', 'no', ':-(', '😢', '😡', 'ugh', 'yuck', 'annoying', 'upset', 'frustrated'}
        
        pos_count = 0
        neg_count = 0
        
        for msg in df['message'].str.lower():
            msg_str = str(msg) if isinstance(msg, str) else ""
            if any(word in msg_str for word in positive_words):
                pos_count += 1
            elif any(word in msg_str for word in negative_words):
                neg_count += 1
        
        total = len(df)
        pos_pct = pos_count / total if total > 0 else 0
        neg_pct = neg_count / total if total > 0 else 0
        neu_pct = 1 - pos_pct - neg_pct
        
        return {
            "positive": max(0, min(1, pos_pct)),
            "neutral": max(0, min(1, neu_pct)),
            "negative": max(0, min(1, neg_pct))
        }
    
    def _calculate_emotions(self, df: pd.DataFrame) -> Dict:
        """Calculate emotion scores using keyword matching"""
        emotions = {
            'joy': ['happy', 'good', 'great', 'awesome', 'love', 'excellent', 'wonderful', 'fantastic', 'nice', 'best', 'perfect', 'cool', 'party', 'fun', 'lol', '😊', '❤️'],
            'anger': ['angry', 'rage', 'hate', 'furious', 'mad', '😡', 'upset', 'mad at'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'fed up', 'sigh', 'ugh', 'seriously'],
            'sadness': ['sad', 'depressed', 'miserable', 'unhappy', '😢', 'sorry', 'miss'],
            'enthusiasm': ['excited', 'amazing', 'awesome', 'great', 'love', 'yes', 'wow'],
            'excitement': ['excited', 'thrilled', 'wow', 'yeah', 'amazing', 'incredible'],
            'concern': ['worry', 'concerned', 'worried', 'care', 'hope', 'please', 'problem']
        }
        
        emotion_scores = {emotion: 0 for emotion in emotions.keys()}
        
        for emotion, keywords in emotions.items():
            count = sum(df['message'].str.lower().str.contains('|'.join(keywords), na=False))
            emotion_scores[emotion] = count / len(df) if len(df) > 0 else 0
        
        # Normalize to sum to 1.0
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        return emotion_scores
    
    def _basic_sentiment_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> Dict:
        """Basic sentiment analysis using keyword matching"""
        positive_words = {'good', 'great', 'awesome', 'amazing', 'happy', 'love', 'excellent', 'wonderful', 'fantastic', 'nice', 'best', 'perfect', 'cool', 'party', 'yes', 'yay', 'lol', ':)', '😊', '❤️'}
        negative_words = {'bad', 'hate', 'angry', 'sad', 'terrible', 'awful', 'horrible', 'worst', 'stupid', 'hate', 'no', ':-(', '😢', '😡', 'ugh', 'yuck'}
        
        pos_count = 0
        neg_count = 0
        
        for msg in df['message'].str.lower():
            msg_str = str(msg) if isinstance(msg, str) else ""
            if any(word in msg_str for word in positive_words):
                pos_count += 1
            elif any(word in msg_str for word in negative_words):
                neg_count += 1
        
        total = len(df)
        pos_pct = pos_count / total if total > 0 else 0
        neg_pct = neg_count / total if total > 0 else 0
        neu_pct = 1 - pos_pct - neg_pct
        
        return {
            "date_range": {"start": start_date, "end": end_date},
            "overall_sentiment": {
                "positive": max(0, min(1, pos_pct)),
                "neutral": max(0, min(1, neu_pct)),
                "negative": max(0, min(1, neg_pct))
            },
            "emotions": {
                "joy": max(0, min(1, pos_pct * 0.6)),
                "anger": max(0, min(1, neg_pct * 0.7)),
                "frustration": max(0, min(1, neg_pct * 0.3)),
                "sadness": max(0, min(1, neg_pct * 0.2)),
                "enthusiasm": max(0, min(1, pos_pct * 0.4))
            },
            "daily_breakdown": self._get_daily_breakdown(df),
            "insights": f"Sentiment distribution: {pos_pct*100:.1f}% positive, {neu_pct*100:.1f}% neutral, {neg_pct*100:.1f}% negative",
            "total_messages": total,
            "primary_emotion": "joy" if pos_pct > neg_pct else "frustration" if neg_pct > 0 else "neutral"
        }
    
    def analyze_by_user(
        self,
        df: pd.DataFrame,
        user: str,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Analyze sentiment for specific user"""
        df_user = df[df['user'] == user].copy()
        
        # Update dataframe for date range analysis
        result = self.analyze_date_range(df_user, start_date, end_date)
        result["user"] = user
        
        return result
    
    def _get_daily_breakdown(self, df: pd.DataFrame) -> list:
        """Get sentiment breakdown by day"""
        daily_data = []
        
        df_sorted = df.sort_values('date')
        for date, group in df_sorted.groupby('only_date'):
            daily_data.append({
                "date": str(date),
                "message_count": len(group),
                "users_count": group['user'].nunique()
            })
        
        return daily_data