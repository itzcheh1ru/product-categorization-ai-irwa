'''*from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from .schemas import ProductData, ProcessingResult, ErrorResponse, HealthCheck
from backend.agents.orchestractor_agent import OrchestratorAgent

from backend.agents.category_classifier_agent import CategoryClassifierAgent
from backend.agents.attribute_extractor_agent import AttributeExtractorAgent
from backend.agents.tag_generator_agent import TagGeneratorAgent
from backend.core.security import decode_access_token, sanitize_input
from backend.utils.config_loader import load_config
from backend.utils.config_loader import load_config


logger = logging.getLogger(__name__)
router = APIRouter()
config = load_config()

# Initialize agents
orchestrator = OrchestratorAgent()
classifier = CategoryClassifierAgent()
extractor = AttributeExtractorAgent()
tagger = TagGeneratorAgent()

# Dependency for optional authentication
async def get_current_user(token: str = None):
    if config['security']['auth_enabled'] and token:
        payload = decode_access_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    return None

@router.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2023-11-15T10:30:00Z",  # You'd use datetime.now().isoformat()
        "model": config['llm']['model']
    }

@router.post("/orchestrator/process", responses={400: {"model": ErrorResponse}})
async def process_product(product_data: Dict[str, Any], user: dict = Depends(get_current_user)):
    try:
        # Sanitize input
        sanitized_data = {
            k: sanitize_input(v) if isinstance(v, str) else v 
            for k, v in product_data.items()
        }
        
        result = orchestrator.process_product(sanitized_data)
        return result
    except Exception as e:
        logger.error(f"Error processing product: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/classifier/process")
async def classify_product(product_data: ProductData, user: dict = Depends(get_current_user)):
    try:
        sanitized_data = {
            k: sanitize_input(v) if isinstance(v, str) else v 
            for k, v in product_data.dict().items()
        }
        
        result = classifier.classify_product(sanitized_data)
        return result
    except Exception as e:
        logger.error(f"Error classifying product: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/extractor/process")
async def extract_attributes(product_data: ProductData, category: str = None, user: dict = Depends(get_current_user)):
    try:
        sanitized_data = {
            k: sanitize_input(v) if isinstance(v, str) else v 
            for k, v in product_data.dict().items()
        }
        
        result = extractor.extract_attributes(sanitized_data, category)
        return result
    except Exception as e:
        logger.error(f"Error extracting attributes: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/tagger/process")
async def generate_tags(product_data: ProductData, attributes: dict = None, user: dict = Depends(get_current_user)):
    try:
        sanitized_data = {
            k: sanitize_input(v) if isinstance(v, str) else v 
            for k, v in product_data.dict().items()
        }
        
        result = tagger.generate_tags(sanitized_data, attributes)
        return result
    except Exception as e:
        logger.error(f"Error generating tags: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/categories")
async def get_categories(user: dict = Depends(get_current_user)):
    return {"categories": classifier.categories}'''

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from backend.agents.orchestractor_agent import OrchestratorAgent
from backend.agents.category_classifier_agent import CategoryClassifierAgent
from backend.agents.attribute_extractor_agent import AttributeExtractorAgent
from backend.agents.tag_generator_agent import TagGeneratorAgent
from backend.core.security import decode_access_token, sanitize_input
from backend.utils.config_loader import load_config
from backend.core.information_retrieval import InformationRetrieval

from .schemas import ProductData, ProcessingResult, ErrorResponse, HealthCheck

logger = logging.getLogger(__name__)
router = APIRouter()
config = load_config()

# -----------------------------
# Initialize Information Retrieval
# -----------------------------
ir_instance = InformationRetrieval(data_path="backend/data/cleaned_product_data.csv")

# -----------------------------
# Initialize agents
# -----------------------------
orchestrator = OrchestratorAgent(ir_instance=ir_instance)
classifier = CategoryClassifierAgent()
extractor = AttributeExtractorAgent()
tagger = TagGeneratorAgent()

# -----------------------------
# Dependency for optional authentication
# -----------------------------
async def get_current_user(token: str = None):
    if config['security'].get('auth_enabled', False) and token:
        payload = decode_access_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    return None

# -----------------------------
# Health check endpoint
# -----------------------------
@router.get("/health", response_model=HealthCheck)
async def health_check():
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": config['llm']['model']
    }

# -----------------------------
# Orchestrator endpoint
# -----------------------------
@router.post("/orchestrator/process", responses={400: {"model": ErrorResponse}})
async def process_product(product_data: Dict[str, Any], user: dict = Depends(get_current_user)):
    try:
        sanitized_data = {k: sanitize_input(v) if isinstance(v, str) else v for k, v in product_data.items()}
        result = orchestrator.process_product(sanitized_data)
        return result
    except Exception as e:
        logger.error(f"Error processing product: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Classifier endpoint
# -----------------------------
@router.post("/classifier/process")
async def classify_product(product_data: ProductData, user: dict = Depends(get_current_user)):
    try:
        sanitized_data = {k: sanitize_input(v) if isinstance(v, str) else v for k, v in product_data.dict().items()}
        result = classifier.classify_product(sanitized_data)
        return result
    except Exception as e:
        logger.error(f"Error classifying product: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Attribute extractor endpoint
# -----------------------------
@router.post("/extractor/process")
async def extract_attributes(product_data: ProductData, category: str = None, user: dict = Depends(get_current_user)):
    try:
        sanitized_data = {k: sanitize_input(v) if isinstance(v, str) else v for k, v in product_data.dict().items()}
        result = extractor.extract_attributes(sanitized_data, category)
        return result
    except Exception as e:
        logger.error(f"Error extracting attributes: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Tag generator endpoint
# -----------------------------
@router.post("/tagger/process")
async def generate_tags(product_data: ProductData, attributes: dict = None, user: dict = Depends(get_current_user)):
    try:
        sanitized_data = {k: sanitize_input(v) if isinstance(v, str) else v for k, v in product_data.dict().items()}
        result = tagger.generate_tags(sanitized_data, attributes)
        return result
    except Exception as e:
        logger.error(f"Error generating tags: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Categories endpoint
# -----------------------------
@router.get("/categories")
async def get_categories(user: dict = Depends(get_current_user)):
    return {"categories": classifier.categories}
