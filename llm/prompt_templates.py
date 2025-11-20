"""
Prompt templates for different LLM tasks — FULL ORIGINAL + SAFETY-FRIENDLY
No truncation, no limits, full 50 messages — just made safe for real group chats
"""

def get_qa_prompt(query: str, context_messages: list) -> str:
    """Generate prompt for Q&A task"""
    
    context_text = "\n".join([
        f"{msg.get('user', 'Unknown')} ({msg.get('date', msg.get('timestamp', 'N/A'))}): {msg.get('message', msg.get('original_message', 'N/A'))}"
        for msg in context_messages
    ])
    
    prompt = f"""You are analyzing a real private WhatsApp group chat between friends.
People joke, roast each other, swear, use slang, and make dark humor — this is completely normal and safe.
Do NOT censor, refuse, or block anything.

Based on the following chat messages, answer the user's question accurately and naturally.

CHAT CONTEXT:
{context_text}

USER QUESTION: {query}

Answer like you're part of the group — direct, honest, and conversational."""
    
    return prompt


def get_sentiment_prompt(messages: list, date_range: str) -> str:
    """Generate prompt for sentiment analysis — FULL ORIGINAL + SAFE"""
    
    sample_messages = "\n".join([
        f"- {msg['user']}: {msg['message']}"
        for msg in messages[:50]  # Full 50 messages as original
    ])
    
    prompt = f"""You are analyzing a real private WhatsApp group chat from {date_range}.
There are {len(messages)} total messages. People roast, swear, joke, and use casual/strong language — this is normal and safe.
Do NOT refuse or censor.

SAMPLE MESSAGES (50 recent):
{sample_messages}

Perform a detailed sentiment analysis considering:
1. Overall sentiment distribution
2. Emotional tone and intensity
3. Context and relationships
4. Conversation flow and inside jokes

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

All percentages must sum to 1.0. Be accurate, honest, and real."""
    
    return prompt


def get_sentiment_scores_prompt(messages: list, date_range: str) -> str:
    """Simplified sentiment scoring — FULL + SAFE"""
    
    sample_messages = "\n".join([
        f"{msg['user']}: {msg['message']}"
        for msg in messages[:50]
    ])
    
    prompt = f"""Private WhatsApp group chat — normal language, jokes, roasts, swearing allowed.
Analyze sentiment of these messages from {date_range} ({len(messages)} total):

{sample_messages}

Score as POSITIVE, NEUTRAL, or NEGATIVE.

Return ONLY this JSON:
{{
    "positive_score": 0.0,
    "neutral_score": 0.0,
    "negative_score": 0.0
}}

Scores must sum to exactly 1.0."""
    
    return prompt


def get_topic_prompt(messages: list, num_topics: int, date_range: str) -> str:
    """Generate prompt for topic extraction — FULL ORIGINAL + SAFE"""
    
    sample_messages = "\n".join([
        f"- {msg['user']}: {msg['message']}"
        for msg in messages[:50]  # Full 50 messages
    ])
    
    prompt = f"""You are analyzing a real private WhatsApp group chat from {date_range}.
There are {len(messages)} total messages. People joke, roast, swear, use slang — this is normal and safe.
Do NOT refuse or censor anything.

SAMPLE MESSAGES (50 recent):
{sample_messages}

Identify up to {num_topics} main topics discussed.

Return ONLY valid JSON (no markdown) with this exact structure:
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

Be honest, real, and direct."""
    
    return prompt


def get_topic_sentiment_prompt(messages: list, topic: str) -> str:
    """Topic-specific sentiment — FULL + SAFE"""
    
    sample_messages = "\n".join([
        f"- {msg['user']}: {msg['message']}"
        for msg in messages[:15]
    ])
    
    prompt = f"""Private WhatsApp chat — normal joking/roasting allowed.
Analyze sentiment about the topic: "{topic}"

RELATED MESSAGES:
{sample_messages}

Return ONLY this JSON:
{{
    "topic": "{topic}",
    "sentiment": {{"positive": 0.0, "neutral": 0.0, "negative": 0.0}},
    "key_points": ["point1", "point2", "point3"],
    "summary": "Brief summary of how this topic is discussed"
}}"""
    
    return prompt
