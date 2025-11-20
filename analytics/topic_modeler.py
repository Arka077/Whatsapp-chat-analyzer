"""
Topic Modeling — FINAL BULLETPROOF VERSION
"""
import pandas as pd
import json
import re
from llm.gemini_client import GeminiClient
from llm.prompt_templates import get_topic_prompt


class TopicModeler:
    def __init__(self):
        self.llm = GeminiClient()

    def extract_topics(self, df: pd.DataFrame, start_date: str, end_date: str, num_topics: int = 5) -> dict:
        df = df.copy()
        df = df[df['only_date'].astype(str).between(start_date, end_date)]
        df = df[df['user'] != 'group_notification']
        df = df[~df['message'].isin(['<Media omitted>', ''])]

        if len(df) < 10:
            return {"topics": [], "analysis_summary": "Not enough messages.", "total_messages": len(df)}

        messages = df.to_dict('records')

        try:
            prompt = get_topic_prompt(messages, num_topics, f"{start_date} to {end_date}")
            raw = self.llm.generate_text(prompt)

            # PERFECT JSON EXTRACTOR — handles ```json
            json_str = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_str:
                raise ValueError("No JSON found")

            result = json.loads(json_str.group())

            result["total_messages"] = len(messages)
            result["date_range"] = f"{start_date} to {end_date}"
            return result

        except Exception as e:
            return {
                "topics": [],
                "analysis_summary": f"Gemini returned messy output. Raw: {str(raw)[:300]}...",
                "total_messages": len(messages),
                "error": str(e)
            }