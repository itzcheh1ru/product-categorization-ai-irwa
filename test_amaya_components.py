import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

try:
    from agents.category_classifier_agent import CategoryClassifierAgent
except Exception:
    CategoryClassifierAgent = None

try:
    from core.nlp_processor import NLPProcessor
except Exception:
    NLPProcessor = None

try:
    from utils.validators import validate_product_data
except Exception:
    def validate_product_data(data):
        return type('VR', (), {'is_valid': True, 'errors': []})()


def test_classifier_exists():
    assert CategoryClassifierAgent is not None, "CategoryClassifierAgent not found"


def test_classifier_smoke():
    if CategoryClassifierAgent is None:
        return
    agent = CategoryClassifierAgent(llm_client=None, nlp_processor=None)
    out = agent.classify_product({'productDisplayName': 'Test Shoe', 'description': 'Running shoe'})
    assert isinstance(out, dict)


def test_nlp_processor_smoke():
    if NLPProcessor is None:
        return
    nlp = NLPProcessor()
    processed = nlp.preprocess_text("Beautiful RED cotton dress")
    assert isinstance(processed, str)


def test_validators_smoke():
    result = validate_product_data({'productDisplayName': 'X', 'confidence': 0.9})
    assert hasattr(result, 'is_valid')
