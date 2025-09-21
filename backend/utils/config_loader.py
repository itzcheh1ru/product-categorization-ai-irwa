import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "backend/config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)