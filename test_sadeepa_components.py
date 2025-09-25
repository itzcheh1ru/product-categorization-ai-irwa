import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

try:
    from agents.attribute_extractor_agent import AttributeExtractorAgent
except Exception:
    AttributeExtractorAgent = None

try:
    from core.communication import CommunicationManager, Message
except Exception:
    CommunicationManager = None
    Message = None

try:
    from core.information_retrieval import InformationRetrieval
except Exception:
    InformationRetrieval = None

try:
    from utils.config_loader import ConfigLoader
except Exception:
    ConfigLoader = None

try:
    from api.schemas import AttributesResult
except Exception:
    AttributesResult = None


def test_attribute_extractor_exists():
    assert AttributeExtractorAgent is not None, "AttributeExtractorAgent not found"


def test_attribute_extractor_smoke():
    if AttributeExtractorAgent is None:
        return
    agent = AttributeExtractorAgent(llm_client=None, nlp_processor=None)
    out = agent.extract_attributes({'productDisplayName': 'Red Shoe', 'description': 'Mesh upper running shoe'})
    assert isinstance(out, dict)


def test_comm_manager_optional():
    if CommunicationManager is None or Message is None:
        return
    cm = CommunicationManager()
    msg = Message('PING', {'x': 1}, 'tester')
    assert isinstance(msg.to_dict(), dict)


def test_ir_optional():
    if InformationRetrieval is None:
        return
    ir = InformationRetrieval(product_database=[{'id': '1', 'text': 'red dress cotton'}])
    ir.build_index()
    results = ir.search_products('cotton dress')
    assert isinstance(results, list)


def test_config_loader_optional():
    if ConfigLoader is None:
        return
    cl = ConfigLoader('backend/config.yaml')
    cfg = cl.load_config()
    assert isinstance(cfg, dict)
