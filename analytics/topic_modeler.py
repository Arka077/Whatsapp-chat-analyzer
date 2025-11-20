"""
Topic Modeling using Gemini — FINAL 100% WORKING
"""
import pandas as pd
from llm.gemini_client import GeminiClient
from llm.prompt_templates import get_topic_prompt


class TopicModeler:
    def __init__(self):
        self.llm = GeminiClient()

    def extract_topics(self, df: pd.DataFrame, start_date: str, end_date: str, num_topics: int = 5) -> dict:
        # Filter messages properly (ONE LINE PER CONDITION)
        df_filtered = df.copy()
        df_filtered = df_filtered[df_filtered['only_date'].astype(str) >= start_date]
        df_filtered = df_filtered[df_filtered['only_date'].astype(str) <= end_date]
        df_filtered = df_filtered[df_filtered['user'] != 'group_notification']
        df_filtered = df_filtered[df_filtered['message'] != '<Media omitted>']
        df_filtered = df_filtered[df_filtered['message'].str.strip() != '']

        if len(df_filtered) < 10:
            return {
                "topics": [],
                "analysis_summary": "Not enough messages to extract topics.",
                "total_messages": len(df_filtered)
            }

        messages = df_filtered.to_dict('records')
        date_range = f"{start_date} to {end_date}"

        try:
            prompt = get_topic_prompt(messages, num_topics, date_range)
            raw_response = self.llm.generate_text(prompt)  # Use generate_text, not generate_json

            # Extract JSON from response (Gemini loves wrapping in ```json)
            import json
            import re
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback: try to parse whole thing
                result = json.loads(raw_response)

            result["total_messages"] = len(messages)
            result["date_range"] = date_range
            return result

        except Exception as e:
            print(f"[TopicModeler] Gemini failed: {e}")
            return {
                "topics": [],
                "analysis_summary": f"Gemini said: '{str(e)[:100]}' — but we're still here!",
                "total_messages": len(messages),
                "fallback": True
            }