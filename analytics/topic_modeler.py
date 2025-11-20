"""
Topic Modeling using Gemini API
"""

import pandas as pd
from typing import Dict, List, Optional
from llm.gemini_client import GeminiClient
from llm.prompt_templates import get_topic_prompt, get_topic_sentiment_prompt
import json

class TopicModeler:
    """Extract and analyze topics from conversations"""
    
    def __init__(self):
        """Initialize topic modeler"""
        self.llm = GeminiClient()
    
    def extract_topics(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        num_topics: int = 5
    ) -> Dict:
        """
        Extract topics from messages in date range
        
        Args:
            df: preprocessed chat dataframe
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            num_topics: number of topics to extract
        
        Returns:
            Dict with extracted topics and analysis
        """
        try:
            # Filter messages
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
                    "topics": [],
                    "analysis_summary": "No messages found in this date range.",
                    "total_messages": 0
                }
            
            # Get topics from Gemini
            date_range_str = f"{start_date} to {end_date}"
            prompt = get_topic_prompt(df_filtered.to_dict('records'), num_topics, date_range_str)
            
            response = self.llm.generate_json(prompt)
            
            # Validate response structure
            if response and isinstance(response, dict) and 'topics' in response:
                topics = response.get('topics', [])
                if isinstance(topics, list) and len(topics) > 0:
                    # Validate at least one topic has required fields
                    if isinstance(topics[0], dict) and 'topic_name' in topics[0]:
                        response['date_range'] = {"start": start_date, "end": end_date}
                        response['total_messages'] = len(df_filtered)
                        return response
            
            # Fallback: Use keyword extraction if Gemini returns invalid response
            print(f"[DEBUG] Gemini returned invalid topic response, using keyword analysis. Response: {response}")
            return self._extract_topics_from_keywords(df_filtered, num_topics, start_date, end_date)
        
        except Exception as e:
            print(f"[DEBUG] Topic extraction error: {str(e)}")
            return self._extract_topics_from_keywords(df, num_topics, start_date, end_date)
    
    def _extract_topics_from_keywords(self, df: pd.DataFrame, num_topics: int, start_date: str, end_date: str) -> Dict:
        """Extract topics using keyword analysis as fallback"""
        from collections import Counter
        
        # Common topic keywords
        topic_keywords = {
            "Work/Study": ["work", "office", "job", "exam", "test", "project", "assignment", "deadline", "boss", "class", "meeting", "report"],
            "Social/Hangout": ["party", "hangout", "meet", "friends", "gather", "group", "plan", "weekend", "fun", "event"],
            "Food/Dining": ["food", "eat", "lunch", "dinner", "breakfast", "restaurant", "cook", "drink", "snack", "pizza", "coffee"],
            "Travel": ["trip", "travel", "hotel", "flight", "visit", "road", "drive", "journey", "tour", "vacation"],
            "Entertainment": ["movie", "watch", "show", "series", "game", "play", "music", "video", "youtube", "netflix", "gaming"],
            "Relationships": ["love", "dating", "relationship", "boyfriend", "girlfriend", "family", "parents", "crush", "dating"]
        }
        
        topics = []
        for topic_id, (topic_name, keywords) in enumerate(topic_keywords.items()):
            matching_messages = df[df['message'].str.lower().str.contains('|'.join(keywords), na=False)]
            
            if len(matching_messages) > 0:
                # Calculate sentiment for this topic
                positive_keywords = {'good', 'great', 'awesome', 'love', 'happy', 'excellent', 'nice', 'cool', 'perfect', 'fun'}
                negative_keywords = {'bad', 'hate', 'hate', 'angry', 'sad', 'terrible', 'awful', 'worst', 'annoying', 'boring'}
                
                pos_count = sum(matching_messages['message'].str.lower().str.contains('|'.join(positive_keywords), na=False))
                neg_count = sum(matching_messages['message'].str.lower().str.contains('|'.join(negative_keywords), na=False))
                
                total_topic = len(matching_messages)
                pos_pct = pos_count / total_topic if total_topic > 0 else 0
                neg_pct = neg_count / total_topic if total_topic > 0 else 0
                neu_pct = max(0, 1 - pos_pct - neg_pct)
                
                topics.append({
                    "topic_name": topic_name,
                    "description": f"Conversation about {topic_name.lower()} - {total_topic} messages",
                    "participants": matching_messages['user'].unique().tolist(),
                    "message_count": total_topic,
                    "sentiment": {
                        "positive": max(0, min(1, pos_pct)),
                        "neutral": max(0, min(1, neu_pct)),
                        "negative": max(0, min(1, neg_pct))
                    },
                    "key_keywords": keywords[:3]
                })
        
        return {
            "date_range": {"start": start_date, "end": end_date},
            "topics": topics[:num_topics] if topics else [
                {
                    "topic_name": "General Chat",
                    "description": "General conversation",
                    "participants": df['user'].unique().tolist(),
                    "message_count": len(df),
                    "sentiment": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                    "key_keywords": ["chat", "message", "discuss"]
                }
            ],
            "analysis_summary": f"Extracted {len(topics)} topics from {len(df)} messages using keyword analysis",
            "total_messages": len(df)
        }
    
    def analyze_topic_sentiment(
        self,
        df: pd.DataFrame,
        topic: str,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Analyze sentiment for specific topic
        
        Args:
            df: dataframe
            topic: topic name
            start_date: start date
            end_date: end date
        
        Returns:
            Topic sentiment analysis
        """
        try:
            # Filter messages
            df_filtered = df[
                (df['only_date'].astype(str) >= start_date) &
                (df['only_date'].astype(str) <= end_date)
            ].copy()
            
            if len(df_filtered) == 0:
                return {
                    "topic": topic,
                    "sentiment": {"positive": 0, "neutral": 1, "negative": 0},
                    "key_points": [],
                    "summary": "No messages found"
                }
            
            prompt = get_topic_sentiment_prompt(df_filtered.to_dict('records'), topic)
            response = self.llm.generate_json(prompt)
            
            return response if response else {
                "topic": topic,
                "sentiment": {"positive": 0, "neutral": 1, "negative": 0},
                "key_points": [],
                "summary": "Error analyzing topic sentiment"
            }
        
        except Exception as e:
            return {
                "topic": topic,
                "error": str(e),
                "sentiment": {"positive": 0, "neutral": 1, "negative": 0},
                "key_points": [],
                "summary": f"Error: {str(e)}"
            }