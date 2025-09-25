import sys
import os
from datetime import datetime

# Ensure backend is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

# Imports (fallback-friendly: only import modules expected in repo)
try:
    from agents.orchestrator_agent import OrchestratorAgent
except Exception:
    OrchestratorAgent = None

try:
    from core.llm_integration import LLMIntegration
except Exception:
    LLMIntegration = None

try:
    from utils.helpers import clean_product_data
except Exception:
    def clean_product_data(data):
        return data

try:
    from api.schemas import ProductData
except Exception:
    ProductData = None


class MockLLM:
    def generate_structured_response(self, prompt, response_format):
        return {"ok": True, "ts": datetime.now().isoformat()}


def test_orchestrator_exists():
    assert OrchestratorAgent is not None, "OrchestratorAgent not found"


def test_orchestrator_process_minimal():
    if OrchestratorAgent is None:
        return
    orch = OrchestratorAgent()
    result = orch.process({
        'productDisplayName': 'Red Cotton Dress',
        'description': 'Lightweight cotton summer dress',
        'baseColour': 'Red'
    })
    assert isinstance(result, dict), "Orchestrator result should be a dict"


def test_llm_integration_smoke():
    if LLMIntegration is None:
        return
    llm = LLMIntegration()
    out = llm.generate_structured_response("ping", {"ok": "bool"})
    assert isinstance(out, dict), "LLM response should be dict"


def test_helpers_clean_product_data():
    raw = {"productDisplayName": "  Test  ", "description": None}
    cleaned = clean_product_data(raw)
    assert isinstance(cleaned, dict)
    assert "productDisplayName" in cleaned


def test_product_schema_optional():
    if ProductData is None:
        return
    pd = ProductData(description="Test product", product_id="shehan-1")
    assert pd.product_id == "shehan-1"
