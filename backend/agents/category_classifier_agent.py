from typing import Dict, Any
import logging
from ..core.llm_integration import LLMIntegration
from ..core.nlp_processor import NLPProcessor
from ..core.information_retrieval import InformationRetrieval

logger = logging.getLogger(__name__)

class CategoryClassifierAgent:
    def __init__(self):
        self.llm = LLMIntegration()
        self.nlp = NLPProcessor()
        self.ir = InformationRetrieval()
        self.categories = ["Apparel", "Accessories", "Footwear", "Personal Care", "Free Items"]
    
    def classify_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            product_name = product_data.get('productDisplayName', '')
            product_description = product_data.get('description', '') or product_name
            
            # Preprocess text
            processed_text = self.nlp.preprocess_text(product_description)
            
            # Get similar products for context
            similar_products = self.ir.search_similar_products(product_name, top_k=3)
            similar_categories = list(set([p.get('masterCategory', '') for p in similar_products]))
            
            # Prepare prompt for LLM
            prompt = f"""
            Classify the following product into one of these categories: {', '.join(self.categories)}
            
            Product: {product_name}
            Description: {product_description}
            
            Similar products are categorized as: {', '.join(similar_categories) if similar_categories else 'No similar products found'}
            
            Respond with a JSON object containing:
            - category: the main category
            - subcategory: a specific subcategory
            - confidence: your confidence level (0.0 to 1.0)
            - reasoning: brief explanation of your classification
            """
            
            response_format = {
                "category": "string",
                "subcategory": "string", 
                "confidence": "float",
                "reasoning": "string"
            }
            
            result = self.llm.generate_structured_response(prompt, response_format)
            
            # Validate and ensure confidence is float
            if 'confidence' in result:
                try:
                    result['confidence'] = float(result['confidence'])
                except (ValueError, TypeError):
                    result['confidence'] = 0.5
            
            return result
            
        except Exception as e:
            logger.error(f"Error in category classification: {e}")
            return {"error": str(e), "category": "Unknown", "subcategory": "Unknown", "confidence": 0.0}