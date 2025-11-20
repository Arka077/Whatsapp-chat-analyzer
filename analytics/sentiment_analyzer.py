"""
Time-based Sentiment Analysis using Gemini (FULL AI MODE)
"""

import pandas as pd
from typing import Dict
from llm.gemini_client import GeminiClient
from llm.prompt_templates import get_sentiment_prompt
import json

class TimeBasedSentimentAnalyzer:
    def __init__(self):
        self.llm = GeminiClient()

    def analyze_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> Dict:
        df_filtered = df[
            (df['only_date'].astype(str) >= start_date) &
            (df['only_date'].astype(str) <= end_date) &
            (df['user'] != 'group_notification') &
            (df['message'] != '<Media omitted>') &
            (df['message'].str.strip() != '')
        ].copy()

        if len(df_filtered) == 0:
            return {
                "insights": "No messages in this date range.",
                "total_messages": 0,
                "overall_sentiment": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                "emotions": {},
                "primary_emotion": "neutral"
            }

        messages = df_filtered.to_dict('records')
        date_range_str = f"{start_date} to {end_date}"

        try:
            prompt = get_sentiment_prompt(messages, date_range_str)
            response = self.llm.generate_json(prompt)

            if not response or not isinstance(response, dict):
                raise ValueError("Invalid response")

            # Use Gemini's full response
            result = {
                "overall_sentiment": response.get("overall_sentiment", {"positive": 0.4, "neutral": 0.4, "negative": 0.2}),
                "emotions": response.get("emotions", {}),
                "insights": response.get("insights", "Gemini analyzed your chat."),
                "total_messages": len(messages),
                "primary_emotion": response.get("primary_emotion", "neutral")
            }
            return result

        except Exception as e:
            print(f"[Gemini Sentiment Error] {e}")
            return {
                "insights": f"AI analysis failed: {str(e)[:100]}",
                "total_messages": len(messages),
                "overall_sentiment": {"positive": 0.4, "neutral": 0.4, "negative": 0.2},
                "emotions": {"joy": 0.5, "concern": 0.3, "frustration": 0.2},
                "primary_emotion": "concern"
            }

    def analyze_by_user(self, df: pd.DataFrame, user: str, start_date: str, end_date: str) -> Dict:
        user_df = df[df['user'] == user]
        result = self.analyze_date_range(user_df, start_date, end_date)
        result["analysis_type"] = f"User: {user}"
        return result