import ollama
import logging
from typing import Dict, Any, List
from backend.utils.config_loader import load_config


logger = logging.getLogger(__name__)
config = load_config()

class LLMIntegration:
    def __init__(self):
        self.model = config['llm']['model']
        self.temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_tokens']
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': full_prompt
                }],
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )
            
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error in LLM integration: {e}")
            return "Error generating response"
    
    def generate_structured_response(self, prompt: str, output_format: Dict[str, Any]) -> Dict[str, Any]:
        format_instructions = f"Respond in JSON format with the following structure: {output_format}"
        full_prompt = f"{prompt}\n\n{format_instructions}"
        
        response = self.generate_response(full_prompt)
        
        try:
            # Extract JSON from response
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except Exception as e:
            logger.warning("Could not parse JSON response, returning fallback values")
            # Build a fallback matching the requested keys if possible
            fallback: Dict[str, Any] = {}
            try:
                for key, val in (output_format or {}).items():
                    if isinstance(val, dict):
                        # attribute-like nested object
                        fallback[key] = {"value": "Unknown", "confidence": 0.0}
                    elif isinstance(val, str):
                        if key == "confidence":
                            fallback[key] = 0.0
                        else:
                            fallback[key] = "Unknown"
                    else:
                        fallback[key] = None
            except Exception:
                fallback = {"response": response}
            return fallback