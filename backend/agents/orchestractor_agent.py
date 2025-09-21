'''from typing import Dict, Any
import logging
from ..core.communication import AgentCommunication
from ..core.llm_integration import LLMIntegration
from ..core.information_retrieval import InformationRetrieval

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self):
        self.communication = AgentCommunication()
        self.llm = LLMIntegration()
        self.ir = InformationRetrieval()
    
    def process_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Step 1: Classify the product
            classification_result = self.communication.send_message(
                "classifier", 
                {"product_data": product_data}
            )
            
            # Step 2: Extract attributes
            extraction_result = self.communication.send_message(
                "extractor", 
                {"product_data": product_data, "category": classification_result.get('category')}
            )
            
            # Step 3: Generate tags
            tagging_result = self.communication.send_message(
                "tagger", 
                {"product_data": product_data, "attributes": extraction_result.get('attributes', {})}
            )
            
            # Step 4: Compile final result
            result = {
                "classification": classification_result,
                "attributes": extraction_result,
                "tags": tagging_result,
                "similar_products": self.ir.search_similar_products(
                    product_data.get('productDisplayName', ''), top_k=3
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            return {"error": str(e)}'''






'2nd*'

'''from typing import Dict, Any
import logging
from backend.core.communication import AgentCommunication
from backend.core.llm_integration import LLMIntegration
from backend.core.information_retrieval import InformationRetrieval

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self, ir_instance: InformationRetrieval):
        self.communication = AgentCommunication()
        self.llm = LLMIntegration()
        self.ir = ir_instance  # Use the IR instance passed from main.py

    def process_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Step 1: Classify the product
            classification_result = self.communication.send_message(
                "classifier", 
                {"product_data": product_data}
            )
            
            # Step 2: Extract attributes
            extraction_result = self.communication.send_message(
                "extractor", 
                {"product_data": product_data, "category": classification_result.get('category')}
            )
            
            # Step 3: Generate tags
            tagging_result = self.communication.send_message(
                "tagger", 
                {"product_data": product_data, "attributes": extraction_result.get('attributes', {})}
            )
            
            # Step 4: Compile final result
            result = {
                "classification": classification_result,
                "attributes": extraction_result,
                "tags": tagging_result,
                "similar_products": self.ir.search_similar_products(
                    product_data.get('productDisplayName', ''), top_k=3
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            return {"error": str(e)}
'''

from typing import Dict, Any
import logging
from backend.core.communication import AgentCommunication
from backend.core.llm_integration import LLMIntegration
from backend.core.information_retrieval import InformationRetrieval

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self, ir_instance: InformationRetrieval):
        self.communication = AgentCommunication()
        self.llm = LLMIntegration()
        self.ir = ir_instance

    def process_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Step 1: Classify the product
            classification_result = self.communication.send_message(
                "classifier", 
                {"product_data": product_data}
            )
            
            # Validate classification result
            if 'error' in classification_result:
                classification_result = {
                    "category": "Unknown",
                    "subcategory": "Unknown",
                    "confidence": 0.0,
                    "reasoning": "Classification failed"
                }
            
            # Step 2: Extract attributes
            extraction_result = self.communication.send_message(
                "extractor", 
                {"product_data": product_data, "category": classification_result.get('category')}
            )
            
            # Get attributes or use defaults
            attributes_data = extraction_result.get('attributes', {}) if 'error' not in extraction_result else {
                "color": {"value": "Unknown", "confidence": 0.0},
                "material": {"value": "Unknown", "confidence": 0.0},
                "size": {"value": "Unknown", "confidence": 0.0},
                "pattern": {"value": "Unknown", "confidence": 0.0},
                "style": {"value": "Unknown", "confidence": 0.0},
                "gender": {"value": "Unknown", "confidence": 0.0},
                "seasonality": {"value": "Unknown", "confidence": 0.0},
                "occasion": {"value": "Unknown", "confidence": 0.0}
            }
            
            # Step 3: Generate tags
            tagging_result = self.communication.send_message(
                "tagger", 
                {"product_data": product_data, "attributes": attributes_data}
            )
            
            # Get tags or use empty list
            tags_data = tagging_result.get('tags', []) if 'error' not in tagging_result else []
            
            # Step 4: Compile final result
            result = {
                "classification": classification_result,
                "attributes": attributes_data,
                "tags": tags_data,
                "similar_products": self.ir.search_similar_products(
                    product_data.get('productDisplayName', ''), top_k=3
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            # Return a valid response structure even on error
            return {
                "classification": {
                    "category": "Error",
                    "subcategory": "Error",
                    "confidence": 0.0,
                    "reasoning": str(e)
                },
                "attributes": {
                    "color": {"value": "Error", "confidence": 0.0},
                    "material": {"value": "Error", "confidence": 0.0},
                    "size": {"value": "Error", "confidence": 0.0},
                    "pattern": {"value": "Error", "confidence": 0.0},
                    "style": {"value": "Error", "confidence": 0.0},
                    "gender": {"value": "Error", "confidence": 0.0},
                    "seasonality": {"value": "Error", "confidence": 0.0},
                    "occasion": {"value": "Error", "confidence": 0.0}
                },
                "tags": [],
                "similar_products": []
            }