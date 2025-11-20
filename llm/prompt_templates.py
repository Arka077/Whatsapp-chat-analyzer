"""
Prompt templates for different LLM tasks
"""

def get_qa_prompt(query: str, context_messages: list) -> str:
    """Generate prompt for Q&A task"""
    
    context_text = "\n".join([
        f"{msg.get('user', 'Unknown')} ({msg.get('date', msg.get('timestamp', 'N/A'))}): {msg.get('message', msg.get('original_message', 'N/A'))}"
        for i, msg in enumerate(context_messages)
    ])
    
    prompt = f"""You are a helpful assistant analyzing WhatsApp chat conversations.

Based on the following chat messages, answer the user's question accurately and concisely.

IMPORTANT INSTRUCTIONS:
- Answer directly without referencing message numbers or citing specific messages
- Provide a natural, conversational response
- Do not mention "According to message X" or similar phrases
- Focus on the information content, not the structure

CHAT CONTEXT:
{context_text}

USER QUESTION: {query}

Provide a direct answer based on the provided chat context. Be specific and natural in your response."""
    
    return prompt


def get_sentiment_prompt(messages: list, date_range: str) -> str:
    """Generate prompt for sentiment analysis"""
    
    sample_messages = "\n".join([
        f"- {msg['user']}: {msg['message']}"
        for msg in messages[:30]  # First 30 messages as sample
    ])
    
    prompt = f"""Analyze the sentiment of these WhatsApp chat messages from {date_range}.

SAMPLE MESSAGES (Total: {len(messages)} messages):
{sample_messages}

Perform a detailed sentiment analysis considering:
1. Overall sentiment distribution across all messages
2. Emotional tone and intensity
3. Context and relationships between participants
4. Message patterns and conversation flow

Return ONLY valid JSON (no markdown, no extra text) with this exact structure:
{{
    "overall_sentiment": {{
        "positive": 0.0,
        "neutral": 0.0,
        "negative": 0.0
    }},
    "emotions": {{
        "joy": 0.0,
        "anger": 0.0,
        "frustration": 0.0,
        "sadness": 0.0,
        "enthusiasm": 0.0,
        "excitement": 0.0,
        "concern": 0.0
    }},
    "primary_emotion": "joy",
    "insights": "Detailed insights about the conversation sentiment and tone"
}}

IMPORTANT:
- All percentages in "overall_sentiment" must sum to 1.0
- All percentages in "emotions" must sum to 1.0
- Return ONLY the JSON object, nothing else
- Be accurate and specific based on actual message content"""
    
    return prompt


def get_sentiment_scores_prompt(messages: list, date_range: str) -> str:
    """Generate prompt specifically for scoring positive/neutral/negative sentiment
    
    This is a simplified prompt focused only on the three main sentiment categories.
    """
    
    # Use all messages for better context
    sample_messages = "\n".join([
        f"{msg['user']}: {msg['message']}"
        for msg in messages[:50]
    ])
    
    prompt = f"""You are analyzing WhatsApp chat messages from {date_range}.
Total messages: {len(messages)}

MESSAGES:
{sample_messages}

Score each message's sentiment as POSITIVE, NEUTRAL, or NEGATIVE based on:
- Positive: Happy, enthusiastic, supportive, grateful, excited
- Neutral: Informational, factual, no emotional content
- Negative: Angry, frustrated, sad, disappointed, critical

Calculate the percentage distribution:

Return ONLY this JSON format (no markdown, no explanation):
{{
    "positive_score": 0.0,
    "neutral_score": 0.0,
    "negative_score": 0.0
}}

Where each score is between 0.0 and 1.0, and they sum to exactly 1.0."""
    
    return prompt


def get_topic_prompt(messages: list, num_topics: int, date_range: str) -> str:
    """Generate prompt for topic extraction"""
    
    sample_messages = "\n".join([
        f"- {msg['user']}: {msg['message']}"
        for msg in messages[:50]  # First 50 messages as sample
    ])
    
    prompt = f"""Identify and extract the main topics discussed in these WhatsApp chat messages from {date_range}.

SAMPLE MESSAGES (Total: {len(messages)} messages):
{sample_messages}

Identify up to {num_topics} distinct topics that were discussed. For each topic:
1. Determine what the conversation was about
2. Identify which participants were involved
3. Estimate the sentiment around that topic
4. Provide a brief description of the discussion

Return ONLY valid JSON (no markdown, no extra text) with this exact structure:
{{
    "topics": [
        {{
            "topic_name": "Topic title (e.g., 'Work Projects', 'Weekend Plans')",
            "description": "What was discussed about this topic",
            "participants": ["Name1", "Name2"],
            "message_count": 0,
            "sentiment": {{
                "positive": 0.0,
                "neutral": 0.0,
                "negative": 0.0
            }},
            "key_keywords": ["keyword1", "keyword2", "keyword3"]
        }}
    ],
    "analysis_summary": "Overall summary of topics discussed and main themes"
}}

IMPORTANT:
- Extract exactly {num_topics} topics (or fewer if fewer distinct topics exist)
- Include realistic participant names from the actual messages
- Each topic's sentiment percentages must sum to 1.0
- Base analysis on actual message content, not assumptions
- Return ONLY the JSON object, nothing else
- Focus on what was actually discussed in the messages"""
    
    return prompt


def get_topic_sentiment_prompt(messages: list, topic: str) -> str:
    """Generate prompt for topic-specific sentiment analysis"""
    
    sample_messages = "\n".join([
        f"- {msg['user']}: {msg['message']}"
        for msg in messages[:15]
    ])
    
    prompt = f"""Analyze sentiment specifically about the topic: "{topic}"

RELATED MESSAGES:
{sample_messages}

Provide sentiment analysis in JSON format:
{{
    "topic": "{topic}",
    "sentiment": {{"positive": 0.0, "neutral": 0.0, "negative": 0.0}},
    "key_points": ["point1", "point2", "point3"],
    "summary": "Brief summary of how this topic is discussed"
}}"""
    
    return prompt