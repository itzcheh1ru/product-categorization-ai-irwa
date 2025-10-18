from typing import Dict, Any

from agents.category_classifier_agent import CategoryClassifierAgent  # type: ignore
from agents.attribute_extractor_agent import AttributeExtractorAgent  # type: ignore
from agents.tag_generator_agent import TagGeneratorAgent
from core.llm_integration import LLMIntegration


class OrchestratorAgent:
    """Coordinates multiple agents to process a product payload."""

    def __init__(self):
        self.classifier = CategoryClassifierAgent()
        self.extractor = AttributeExtractorAgent()
        self.tagger = TagGeneratorAgent()

    def process(self, product: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"input": product}

        attributes = self.extractor.extract_attributes(product)
        result["attributes"] = attributes

        classification = self.classifier.classify_product(product)
        result["classification"] = classification

        tags = self.tagger.generate_tags(product, attributes.get("attributes"))
        result["tags"] = tags

        return result


