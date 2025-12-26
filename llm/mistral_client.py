"""
Mistral AI client for LLM operations
WITH MULTI-API KEY FALLBACK SUPPORT
"""

from mistralai import Mistral
from typing import Optional, Dict, Any, List
import json
import time

# Import config values that are SAFE at import time
from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P
from config.settings import get_mistral_api_keys


class MistralClient:
    """Wrapper around Mistral API with multi-key fallback support"""
    
    def __init__(self):
        """Initialize Mistral client with multiple API keys for fallback"""
        # Get all available API keys
        self.api_keys = get_mistral_api_keys()
        self.current_key_index = 0
        
        # Configure with first key
        self._configure_api(self.api_keys[self.current_key_index])
        
        self.model = LLM_MODEL
        self. temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.top_p = LLM_TOP_P
    
    def _configure_api(self, api_key: str):
        """Configure Mistral API with specific key"""
        self. client = Mistral(api_key=api_key)
        print(f"[Mistral] Configured with API key #{self. current_key_index + 1}")
    
    def _switch_to_next_key(self) -> bool:
        """
        Switch to next available API key
        
        Returns:
            True if switched successfully, False if no more keys
        """
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self._configure_api(self.api_keys[self.current_key_index])
            print(f"[Mistral] Switched to API key #{self.current_key_index + 1}")
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
            "timeout",
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
                
                print(f"[Mistral] Error with key #{self.current_key_index + 1}: {error_msg[: 100]}")
                
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
                            self._configure_api(self. api_keys[self.current_key_index])
                            time.sleep(1)
                            continue
                        else: 
                            # All keys tried, raise error
                            break
                else:
                    # Non-retryable error
                    raise
        
        # All retries exhausted
        if last_error:
            raise last_error
    
    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Generate text with automatic API key fallback"""
        
        def _generate():
            temp = temperature if temperature is not None else self.temperature
            
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temp,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            
            return ""
        
        try:
            return self._execute_with_fallback(_generate)
        
        except Exception as e:
            error_msg = str(e)
            print(f"[Mistral Error] All API keys exhausted: {e}")
            return f"[Error:  All API keys failed.  {error_msg[: 100]}]"
    
    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]: 
        """Generate JSON response with automatic API key fallback"""
        
        def _generate():
            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nRespond with valid JSON only."
            response_text = self.generate_text(json_prompt)
            
            # Check if response is an error
            if response_text. startswith("[Error"):
                print(f"[Mistral] Error response: {response_text}")
                return None
            
            if not response_text. strip():
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
            print(f"[Mistral Error] JSON decode failed: {e}")
            return None
        except Exception as e: 
            print(f"[Mistral Error] generate_json failed: {e}")
            return None
