import json
import logging
from typing import Any, Dict
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON string, returning empty dict on failure"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}