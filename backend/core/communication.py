'''*import json
import requests
from typing import Dict, Any, List
from backend.utils.config_loader import load_config


config = load_config()

class AgentCommunication:
    def __init__(self):
        self.base_url = f"http://{config['api']['host']}:{config['api']['port']}"
    
    def send_message(self, agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/api/{agent}/process"
            response = requests.post(url, json=message, timeout=config['agents']['orchestrator']['timeout'])
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Communication error with {agent}: {str(e)}"}
    
    def broadcast(self, agents: List[str], message: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for agent in agents:
            results[agent] = self.send_message(agent, message)
        return results
        '''


import json
from typing import Dict, Any, List
from backend.utils.config_loader import load_config
from backend.agents.category_classifier_agent import CategoryClassifierAgent
from backend.agents.attribute_extractor_agent import AttributeExtractorAgent
from backend.agents.tag_generator_agent import TagGeneratorAgent

config = load_config()

class AgentCommunication:
    def __init__(self):
        # Initialize agent instances
        self.classifier = CategoryClassifierAgent()
        self.extractor = AttributeExtractorAgent()
        self.tagger = TagGeneratorAgent()
    
    def send_message(self, agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if agent == "classifier":
                return self.classifier.classify_product(message.get('product_data', {}))
            elif agent == "extractor":
                return self.extractor.extract_attributes(
                    message.get('product_data', {}),
                    message.get('category')
                )
            elif agent == "tagger":
                return self.tagger.generate_tags(
                    message.get('product_data', {}),
                    message.get('attributes')
                )
            else:
                return {"error": f"Unknown agent: {agent}"}
        except Exception as e:
            return {"error": f"Error in {agent}: {str(e)}"}
    
    def broadcast(self, agents: List[str], message: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for agent in agents:
            results[agent] = self.send_message(agent, message)
        return results