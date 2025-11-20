"""
Topic Modeling using Gemini (FULL AI MODE)
"""

import pandas as pd
from typing import Dict
from llm.gemini_client import GeminiClient
from llm.prompt_templates import get_topic_prompt
import json

class TopicModeler:
    def __init__(self):
        self.llm = GeminiClient()

    def extract_topics(self, df: pd.DataFrame, start_date: str, end_date: str, num_topics: int = 5) -> Dict:
        df_filtered = df[
            (df['only_date'].astype(str) >= start_date) &
            (df['only_date'].astype(str) <= end_date) &
            (df['user'] != 'group_notification') &
            (df['message'] != '<Media omitted>') &
            (df['message'].str.strip() != '')
        ].copy()

        if len(df_filtered) < 10:
            return {
                "topics": [],
                "analysis_summary": "Not enough messages for topic modeling.",
                "total_messages": len(df_filtered),
                "date_range": f"{start_date} to {end_date}"
            }

        messages = df_filtered.to_dict('records')
        date_range_str = f"{start_date} to {end_date}"

        try:
            prompt = get_topic_prompt(messages, num_topics, date_range_str)
            response = self.llm.generate_json(prompt)

            if response and isinstance(response, dict) and "topics" in response:
                result = {
                    "topics": response.get("topics", []),
                    "analysis_summary": response.get("analysis_summary", "Gemini extracted topics."),
                    "total_messages": len(messages),
                    "date_range": f"{start_date} to {end_date}"
                }
                return result

        except Exception as e:
            print(f"[Gemini Topic Error] {e}")

        # Final fallback
        return {
            "topics": [],
            "analysis_summary": f"AI topic extraction failed: {str(e)[:100]}",
            "total_messages": len(messages),
            "date_range": f"{start_date} to {end_date}"
        }