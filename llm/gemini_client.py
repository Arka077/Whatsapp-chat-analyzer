"""
Gemini API client for LLM operations — FIXED FOR 2.5 FLASH JSON BUG
"""

import google.generativeai as genai
from typing import Optional, Dict, Any
import json
import re
import os

from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from config.settings import get_gemini_api_key

class GeminiClient:
    """Wrapper around Gemini API — JSON mode for 2.5 Flash bug fix"""
    
    def __init__(self):
        """Initialize Gemini client"""
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)
        
        # Use 2.5 Flash with JSON mode enabled (fixes truncation)
        self.model = genai.GenerativeModel(
            LLM_MODEL,  # Your "gemini-2.5-flash"
            generation_config={
                "temperature": LLM_TEMPERATURE,
                "top_p": 0.9,
                "max_output_tokens": LLM_MAX_TOKENS,  # Keep your 2000 — JSON mode makes it efficient
            },
            # CRITICAL: Force JSON output to avoid token hogging
            system_instruction="You are a helpful assistant. Always respond with valid JSON only."
        )
    
    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        try:
            config = self.generation_config.copy()
            if temperature is not None:
                config["temperature"] = temperature
            
            response = self.model.generate_content(
                prompt,
                # THE FIX: Force JSON mode — no extra text, no truncation
                generation_config=genai.types.GenerationConfig(**config),
                # Enforce JSON output (supported in 2.5 Flash)
                response_mime_type="application/json"
            )
            
            return response.text if response.text else ""
        
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return ""
    
    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate and parse JSON — uses JSON mode to avoid bugs"""
        try:
            response_text = self.generate_text(prompt)
            
            if not response_text:
                return None
            
            # Extract JSON from response (handles any markdown)
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                # Raw JSON
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
            
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Response text: {response_text[:500]}")
            return None
        except Exception as e:
            print(f"Error generating JSON: {str(e)}")
            return None
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        try:
            response = self.model.count_tokens(text)
            return response.total_tokens
        except:
            return len(text) // 4