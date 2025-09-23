from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class LLMIntegration:
    """
    Lightweight LLM wrapper using Ollama (local) if available.
    Fallback: echo structure without real inference.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("LLM_MODEL", "llama3.1")

    def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        try:
            import ollama  # type: ignore
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
            return response.get("message", {}).get("content", "").strip()
        except Exception:
            # Fallback if ollama isn't running
            return ""

    def generate_structured_response(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        text = self.generate_text(prompt)
        if not text:
            # Fallback empty structure
            def _empty(obj):
                if isinstance(obj, dict):
                    return {k: _empty(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return []
                return None

            return _empty(schema)  # type: ignore

        # Best-effort: try to parse JSON if present
        import json
        try:
            return json.loads(text)
        except Exception:
            return {k: [] if isinstance(v, list) else None for k, v in schema.items()}  # type: ignore


