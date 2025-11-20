"""
Gemini API client for LLM operations
SAFE FOR STREAMLIT CLOUD – no import-time crashes
WITH SAFETY SETTINGS TO PREVENT BLOCKS
"""

import google.generativeai as genai
from typing import Optional, Dict, Any
import json

# Import config values that are SAFE at import time
from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from config.settings import get_gemini_api_key  # ← function, not value


class GeminiClient:
    """Wrapper around Gemini API – lazy loads API key safely"""
    
    def __init__(self):
        """Initialize Gemini client – only called when actually used"""
        # CRITICAL: Get the key HERE, not at module level
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(LLM_MODEL)
        self.generation_config = {
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.9,
            "max_output_tokens": LLM_MAX_TOKENS,
        }
        
        # CRITICAL: Set safety settings to allow chat analysis
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
    
    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        try:
            config = self.generation_config.copy()
            if temperature is not None:
                config["temperature"] = temperature
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(**config),
                safety_settings=self.safety_settings  # Add safety settings
            )
            
            # Handle blocked responses gracefully
            if not response.candidates:
                return "[Response blocked by safety filters. Try rephrasing your question or adjusting date range.]"
            
            # Check if response was blocked
            if response.candidates[0].finish_reason == 2:  # SAFETY
                return "[Response blocked due to safety concerns. The content may have triggered filters. Try a different query.]"
            
            return response.text or ""
        
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific safety-related errors
            if "finish_reason" in error_msg or "SAFETY" in error_msg:
                return "[Content was flagged by safety filters. Try rephrasing your question or use a different date range.]"
            
            print(f"[Gemini Error] generate_text: {e}")
            return f"[Error: {error_msg[:150]}]"
    
    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            response_text = self.generate_text(prompt)
            
            # Check if response was blocked
            if response_text.startswith("["):
                print(f"[Gemini] Blocked response: {response_text}")
                return None
            
            if not response_text.strip():
                return None

            # Extract JSON from ```json or ```
            start = response_text.find("```json")
            if start == -1:
                start = response_text.find("```")
                if start != -1:
                    start += 3
            else:
                start += 7

            if start != -1:
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip() if end != -1 else response_text[start:].strip()
            else:
                # Fallback: find first { to last }
                json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]

            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            print(f"[Gemini JSON Parse Error]: {e}")
            print(f"Raw output: {response_text[:500]}")
            return None
        except Exception as e:
            print(f"[Gemini generate_json error]: {e}")
            return None
    
    def count_tokens(self, text: str) -> int:
        try:
            return self.model.count_tokens(text).total_tokens
        except:
            return len(text) // 4  # fallback