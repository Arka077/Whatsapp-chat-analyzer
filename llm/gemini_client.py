"""
Gemini API client for LLM operations
"""

import google.generativeai as genai
from typing import Optional, Dict, Any
import json
from config.settings import  LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from config.settings import get_gemini_api_key
GEMINI_API_KEY = get_gemini_api_key()   # call only when needed

class GeminiClient:
    """Wrapper around Gemini API"""
    
    def __init__(self):
        """Initialize Gemini client"""
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(LLM_MODEL)
        self.generation_config = {
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.9,
            "max_output_tokens": LLM_MAX_TOKENS,
        }
    
    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate text using Gemini
        
        Args:
            prompt: input prompt
            temperature: optional override temperature
        
        Returns:
            Generated text
        """
        try:
            config = self.generation_config.copy()
            if temperature is not None:
                config["temperature"] = temperature
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(**config)
            )
            
            return response.text if response.text else ""
        
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return ""
    
    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate JSON response using Gemini
        
        Args:
            prompt: input prompt
        
        Returns:
            Parsed JSON dict
        """
        try:
            response_text = self.generate_text(prompt)
            
            if not response_text or response_text.strip() == "":
                print("[DEBUG] Gemini returned empty response")
                return None
            
            # Extract JSON from response (handle markdown code blocks)
            # Try to find JSON wrapped in markdown code blocks first
            if "```json" in response_text:
                start_idx = response_text.find("```json") + 7
                end_idx = response_text.find("```", start_idx)
                if end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx].strip()
                    return json.loads(json_str)
            elif "```" in response_text:
                start_idx = response_text.find("```") + 3
                end_idx = response_text.find("```", start_idx)
                if end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx].strip()
                    return json.loads(json_str)
            
            # Try to find raw JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                return parsed
            
            print(f"[DEBUG] Could not extract JSON from response: {response_text[:200]}")
            return None
        
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {str(e)}")
            print(f"[DEBUG] Response text: {response_text[:500]}")
            return None
        except Exception as e:
            print(f"[DEBUG] Error generating JSON: {str(e)}")
            return None
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        try:
            response = self.model.count_tokens(text)
            return response.total_tokens
        except:
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4