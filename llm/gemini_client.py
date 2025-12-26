"""
Gemini API client for LLM operations
SAFE FOR STREAMLIT CLOUD – no import-time crashes
WITH SAFETY SETTINGS TO PREVENT BLOCKS
WITH MULTI-API KEY FALLBACK SUPPORT
UPDATED TO USE NEW google.genai PACKAGE
"""

from google import genai
from google.genai import types
from typing import Optional, Dict, Any, List
import json
import time

# Import config values that are SAFE at import time
from config. settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from config.settings import get_gemini_api_keys  # ← function that returns list


class GeminiClient:
    """Wrapper around Gemini API with multi-key fallback support"""
    
    def __init__(self):
        """Initialize Gemini client with multiple API keys for fallback"""
        # Get all available API keys
        self.api_keys = get_gemini_api_keys()
        self.current_key_index = 0
        
        # Configure with first key
        self._configure_api(self.api_keys[self.current_key_index])
        
        self.generation_config = types.GenerateContentConfig(
            temperature=LLM_TEMPERATURE,
            top_p=0.9,
            max_output_tokens=LLM_MAX_TOKENS,
        )
        
        # CRITICAL: Set safety settings to allow chat analysis
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",
            ),
            types. SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",
            ),
        ]
    
    def _configure_api(self, api_key: str):
        """Configure Gemini API with specific key"""
        self.client = genai.Client(api_key=api_key)
        print(f"[Gemini] Configured with API key #{self.current_key_index + 1}")
    
    def _switch_to_next_key(self) -> bool:
        """
        Switch to next available API key
        
        Returns:
            True if switched successfully, False if no more keys
        """
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self._configure_api(self.api_keys[self.current_key_index])
            print(f"[Gemini] Switched to API key #{self.current_key_index + 1}")
            return True
        return False
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if error is retryable with different API key"""
        retryable_keywords = [
            "quota",
            "rate limit",
            "429",
            "resource exhausted",
            "too many requests",
            "unavailable",
            "503",
            "500",
        ]
        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in retryable_keywords)
    
    def _execute_with_fallback(self, operation_func, max_retries:  int = None):
        """
        Execute operation with automatic API key fallback
        
        Args: 
            operation_func: Function to execute
            max_retries: Max retries per key (default: number of keys)
        
        Returns:
            Result from operation_func
        """
        if max_retries is None: 
            max_retries = len(self.api_keys)
        
        last_error = None
        attempts = 0
        keys_tried = set()
        
        while attempts < max_retries:
            try:
                # Mark this key as tried
                keys_tried.add(self.current_key_index)
                
                # Execute the operation
                result = operation_func()
                return result
                
            except Exception as e:
                error_msg = str(e)
                last_error = e
                attempts += 1
                
                print(f"[Gemini] Error with key #{self.current_key_index + 1}: {error_msg[: 100]}")
                
                # Check if error is retryable
                if self._is_retryable_error(error_msg):
                    # Try to switch to next key
                    if self._switch_to_next_key():
                        # Small delay before retry
                        time.sleep(0.5)
                        continue
                    else:
                        # No more keys, reset to first untried key or first key
                        untried_keys = [i for i in range(len(self.api_keys)) if i not in keys_tried]
                        if untried_keys:
                            self.current_key_index = untried_keys[0]
                            self._configure_api(self.api_keys[self.current_key_index])
                            time.sleep(1)
                            continue
                        else: 
                            # All keys tried, raise error
                            break
                else:
                    # Non-retryable error (like safety filter)
                    raise
        
        # All retries exhausted
        if last_error:
            raise last_error
    
    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Generate text with automatic API key fallback"""
        
        def _generate():
            config = self.generation_config
            if temperature is not None:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=0.9,
                    max_output_tokens=LLM_MAX_TOKENS,
                )
            
            response = self.client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=config,
            )
            
            # Handle blocked responses gracefully
            if not response.candidates:
                return "[Response blocked by safety filters.  Try rephrasing your question or adjusting date range.]"
            
            # Check if response was blocked
            if hasattr(response. candidates[0], 'finish_reason') and response.candidates[0].finish_reason == 2:  # SAFETY
                return "[Response blocked due to safety concerns. The content may have triggered filters.  Try a different query.]"
            
            return response.text or ""
        
        try:
            return self._execute_with_fallback(_generate)
        
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific safety-related errors
            if "finish_reason" in error_msg or "SAFETY" in error_msg: 
                return "[Content was flagged by safety filters. Try rephrasing your question or use a different date range.]"
            
            print(f"[Gemini Error] All API keys exhausted: {e}")
            return f"[Error: All API keys failed. {error_msg[: 100]}]"
    
    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]: 
        """Generate JSON response with automatic API key fallback"""
        
        def _generate():
            response_text = self.generate_text(prompt)
            
            # Check if response was blocked
            if response_text.startswith("["):
                print(f"[Gemini] Blocked response:  {response_text}")
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
                if end != -1:
                    json_str = response_text[start: end]. strip()
                    return json.loads(json_str)

            # Try parsing entire response
            return json.loads(response_text. strip())
        
        try: 
            return self._execute_with_fallback(_generate)
        except json.JSONDecodeError as e:
            print(f"[Gemini Error] JSON decode failed: {e}")
            return None
        except Exception as e: 
            print(f"[Gemini Error] generate_json failed: {e}")
            return None
