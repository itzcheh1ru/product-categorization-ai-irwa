'''*from typing import Dict, Any, List
import logging
from ..core.llm_integration import LLMIntegration
from ..core.nlp_processor import NLPProcessor

logger = logging.getLogger(__name__)

class TagGeneratorAgent:
    def __init__(self):
        self.llm = LLMIntegration()
        self.nlp = NLPProcessor()
    
    def generate_tags(self, product_data: Dict[str, Any], attributes: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            product_name = product_data.get('productDisplayName', '')
            base_color = product_data.get('baseColour', '')
            usage = product_data.get('usage', '')
            
            # Prepare prompt for LLM
            prompt = f"""
            Generate relevant tags for the following product:
            
            Product: {product_name}
            Base Color: {base_color}
            Usage: {usage}
            
            Extracted Attributes: {attributes if attributes else 'No attributes provided'}
            
            Generate 5-10 relevant tags that would help customers find this product.
            Include tags for:
            - Product type
            - Color
            - Style
            - Occasion
            - Material (if known)
            - Season
            
            Respond with a JSON object containing a list of tags with confidence scores.
            """
            
            response_format = {
                "tags": [
                    {"tag": "string", "confidence": "float"}
                ]
            }
            
            result = self.llm.generate_structured_response(prompt, response_format)
            
            # Also generate tags using NLP
            nlp_tags = self.generate_nlp_tags(product_name, attributes)
            if nlp_tags:
                if 'tags' not in result:
                    result['tags'] = []
                result['tags'].extend(nlp_tags)
            
            # Deduplicate and sort by confidence
            if 'tags' in result:
                seen = set()
                unique_tags = []
                for tag_obj in result['tags']:
                    tag_name = tag_obj['tag'].lower().strip()
                    if tag_name not in seen:
                        seen.add(tag_name)
                        unique_tags.append(tag_obj)
                result['tags'] = sorted(unique_tags, key=lambda x: x.get('confidence', 0), reverse=True)[:10]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in tag generation: {e}")
            return {"error": str(e), "tags": []}
    
    def generate_nlp_tags(self, product_name: str, attributes: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        tags = []
        
        # Extract nouns and adjectives
        nouns = self.nlp.extract_nouns(product_name)
        adjectives = self.nlp.extract_adjectives(product_name)
        
        for noun in nouns:
            tags.append({"tag": noun, "confidence": 0.7, "source": "NLP"})
        
        for adj in adjectives:
            tags.append({"tag": adj, "confidence": 0.6, "source": "NLP"})
        
        # Add attribute values as tags
        if attributes:
            for attr_name, attr_value in attributes.items():
                if isinstance(attr_value, dict) and 'value' in attr_value:
                    tag_value = attr_value['value']
                    if tag_value and tag_value not in ["", "Not specified", "Unknown"]:
                        tags.append({
                            "tag": tag_value, 
                            "confidence": attr_value.get('confidence', 0.5) * 0.8,
                            "source": "attribute"
                        })
        
        return tags
        '''


from typing import Dict, Any, List
import logging
import re
from ..core.llm_integration import LLMIntegration
from ..core.nlp_processor import NLPProcessor

logger = logging.getLogger(__name__)

class TagGeneratorAgent:
    def __init__(self):
        self.llm = LLMIntegration()
        self.nlp = NLPProcessor()
    
    def generate_tags(self, product_data: Dict[str, Any], attributes: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            product_name = product_data.get('productDisplayName', '')
            base_color = product_data.get('baseColour', '')
            usage = product_data.get('usage', '')
            
            # Prepare prompt for LLM
            prompt = f"""
            Generate specific, descriptive tags for this product:

            PRODUCT: {product_name}
            BASE COLOR: {base_color}
            USAGE: {usage}

            Create 5-10 specific tags that describe this product accurately.
            Examples of good tags: "cotton-tshirt", "red-dress", "winter-boots", "formal-shirt"
            Examples of bad tags: "product", "color", "style" (too generic)

            Make tags lowercase with hyphens, descriptive and specific.
            """

            response_format = {
                "tags": [
                    {"tag": "string", "confidence": "float"}
                ]
            }
            
            result = self.llm.generate_structured_response(prompt, response_format)
            
            # Also generate tags using NLP
            nlp_tags = self.generate_nlp_tags(product_name, attributes)
            
            # Combine and deduplicate tags
            all_tags = []
            
            # Add LLM tags
            if 'tags' in result and isinstance(result['tags'], list):
                for tag_obj in result['tags']:
                    if isinstance(tag_obj, dict) and 'tag' in tag_obj:
                        tag_text = self.clean_tag(tag_obj['tag'])
                        if tag_text:
                            all_tags.append({
                                "tag": tag_text,
                                "confidence": float(tag_obj.get('confidence', 0.7)),
                                "source": "llm"
                            })
            
            # Add NLP tags
            all_tags.extend(nlp_tags)
            
            # Deduplicate and sort by confidence
            seen = set()
            unique_tags = []
            for tag_obj in all_tags:
                tag_name = tag_obj['tag'].lower().strip()
                if tag_name not in seen and tag_name:
                    seen.add(tag_name)
                    unique_tags.append(tag_obj)
            
            # Limit to top 10 tags
            result['tags'] = sorted(unique_tags, key=lambda x: x.get('confidence', 0), reverse=True)[:10]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in tag generation: {e}")
            return {"tags": []}
    
    def clean_tag(self, tag: str) -> str:
        """Clean and format tag text"""
        if not tag:
            return ""
        
        # Remove generic words
        tag = tag.lower().strip()
        generic_words = ['product', 'type', 'color', 'style', 'occasion', 'material', 'season']
        
        for word in generic_words:
            tag = re.sub(rf'\b{word}\b', '', tag)
        
        # Clean up and format
        tag = re.sub(r'[^\w\s-]', '', tag)  # Remove special chars
        tag = re.sub(r'\s+', '-', tag.strip())  # Replace spaces with hyphens
        tag = re.sub(r'-+', '-', tag)  # Remove multiple hyphens
        
        return tag if tag and tag != '-' else ""
    
    def generate_nlp_tags(self, product_name: str, attributes: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        tags = []
        
        # Extract nouns and adjectives from product name
        nouns = self.nlp.extract_nouns(product_name)
        adjectives = self.nlp.extract_adjectives(product_name)
        
        for noun in nouns:
            clean_noun = self.clean_tag(noun)
            if clean_noun:
                tags.append({"tag": clean_noun, "confidence": 0.7, "source": "nlp"})
        
        for adj in adjectives:
            clean_adj = self.clean_tag(adj)
            if clean_adj:
                tags.append({"tag": clean_adj, "confidence": 0.6, "source": "nlp"})
        
        # Add attribute values as tags
        if attributes:
            for attr_name, attr_value in attributes.items():
                if isinstance(attr_value, dict) and 'value' in attr_value:
                    tag_value = attr_value['value']
                    if tag_value and tag_value not in ["", "Unknown", "Not specified"]:
                        clean_tag = self.clean_tag(tag_value)
                        if clean_tag:
                            tags.append({
                                "tag": clean_tag, 
                                "confidence": attr_value.get('confidence', 0.5) * 0.8,
                                "source": "attribute"
                            })
        
        return tags