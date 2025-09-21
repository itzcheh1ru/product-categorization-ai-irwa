'''*from typing import Dict, Any, List
import logging
from ..core.llm_integration import LLMIntegration
from ..core.nlp_processor import NLPProcessor

logger = logging.getLogger(__name__)

class AttributeExtractorAgent:
    def __init__(self):
        self.llm = LLMIntegration()
        self.nlp = NLPProcessor()
    
    def extract_attributes(self, product_data: Dict[str, Any], category: str = None) -> Dict[str, Any]:
        try:
            product_name = product_data.get('productDisplayName', '')
            base_color = product_data.get('baseColour', '')
            usage = product_data.get('usage', '')
            season = product_data.get('season', '')
            
            # Prepare prompt for LLM
            prompt = f"""
            Extract detailed attributes from the following product information:
            
            Product: {product_name}
            Base Color: {base_color}
            Usage: {usage}
            Season: {season}
            Category: {category if category else 'Not specified'}
            
            Extract the following attributes:
            - color: primary color(s)
            - material: fabric or material composition
            - size: size information if available
            - pattern: pattern or design
            - style: fashion or product style
            - gender: intended gender if specified
            - seasonality: appropriate seasons
            - occasion: suitable occasions
            
            Respond with a JSON object containing these attributes.
            For each attribute, provide a value and confidence score (0.0 to 1.0).
            """
            
            response_format = {
                "color": {"value": "string", "confidence": "float"},
                "material": {"value": "string", "confidence": "float"},
                "size": {"value": "string", "confidence": "float"},
                "pattern": {"value": "string", "confidence": "float"},
                "style": {"value": "string", "confidence": "float"},
                "gender": {"value": "string", "confidence": "float"},
                "seasonality": {"value": "string", "confidence": "float"},
                "occasion": {"value": "string", "confidence": "float"}
            }
            
            result = self.llm.generate_structured_response(prompt, response_format)
            
            # Also extract entities using NLP
            nlp_entities = self.nlp.extract_entities(product_name)
            if nlp_entities:
                result['entities'] = nlp_entities
            
            return {"attributes": result}
            
        except Exception as e:
            logger.error(f"Error in attribute extraction: {e}")
            return {"error": str(e), "attributes": {}}
            '''

from typing import Dict, Any, List
import logging
from ..core.llm_integration import LLMIntegration
from ..core.nlp_processor import NLPProcessor

logger = logging.getLogger(__name__)

class AttributeExtractorAgent:
    def __init__(self):
        self.llm = LLMIntegration()
        self.nlp = NLPProcessor()
    
    def extract_attributes(self, product_data: Dict[str, Any], category: str = None) -> Dict[str, Any]:
        try:
            product_name = product_data.get('productDisplayName', '')
            base_color = product_data.get('baseColour', '')
            usage = product_data.get('usage', '')
            season = product_data.get('season', '')
            
            # Prepare more specific prompt
            prompt = f"""
            Analyze this product and extract SPECIFIC attribute values. Do NOT use generic placeholder text.

            PRODUCT: {product_name}
            BASE COLOR: {base_color}
            USAGE: {usage}
            SEASON: {season}
            CATEGORY: {category if category else 'Not specified'}

            Extract REAL, SPECIFIC values for these attributes. If information is not available, make a reasonable inference or use "Unknown".

            For example:
            - If product is "Red Cotton T-Shirt", color should be "red", material should be "cotton"
            - If product is "Winter Boots", seasonality should be "winter"
            - If product is "Men's Formal Shirt", gender should be "men", style should be "formal"

            Respond with a JSON object containing these attributes with specific, real values.
            """

            response_format = {
                "color": {"value": "string", "confidence": "float"},
                "material": {"value": "string", "confidence": "float"},
                "size": {"value": "string", "confidence": "float"},
                "pattern": {"value": "string", "confidence": "float"},
                "style": {"value": "string", "confidence": "float"},
                "gender": {"value": "string", "confidence": "float"},
                "seasonality": {"value": "string", "confidence": "float"},
                "occasion": {"value": "string", "confidence": "float"}
            }
            
            result = self.llm.generate_structured_response(prompt, response_format)
            
            # Post-process to ensure valid values
            for attr_name, attr_data in result.items():
                if isinstance(attr_data, dict):
                    # Clean up values
                    value = attr_data.get('value', '').strip()
                    if not value or value.lower() in ['', 'not specified', 'unknown', 'n/a', 'none']:
                        attr_data['value'] = "Unknown"
                    
                    # Ensure confidence is valid float
                    try:
                        confidence = float(attr_data.get('confidence', 0.5))
                        attr_data['confidence'] = max(0.0, min(1.0, confidence))
                    except (ValueError, TypeError):
                        attr_data['confidence'] = 0.5
            
            # Also extract entities using NLP
            nlp_entities = self.nlp.extract_entities(product_name)
            if nlp_entities:
                result['entities'] = nlp_entities
            
            return {"attributes": result}
            
        except Exception as e:
            logger.error(f"Error in attribute extraction: {e}")
            # Return default attributes structure instead of error
            default_attrs = {
                "color": {"value": "Unknown", "confidence": 0.0},
                "material": {"value": "Unknown", "confidence": 0.0},
                "size": {"value": "Unknown", "confidence": 0.0},
                "pattern": {"value": "Unknown", "confidence": 0.0},
                "style": {"value": "Unknown", "confidence": 0.0},
                "gender": {"value": "Unknown", "confidence": 0.0},
                "seasonality": {"value": "Unknown", "confidence": 0.0},
                "occasion": {"value": "Unknown", "confidence": 0.0}
            }
            return {"attributes": default_attrs}