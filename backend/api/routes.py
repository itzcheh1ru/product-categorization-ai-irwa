from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from ..agents.category_classifier_agent import CategoryClassifierAgent  # type: ignore
from ..agents.attribute_extractor_agent import AttributeExtractorAgent  # type: ignore
from ..agents.tag_generator_agent import TagGeneratorAgent
from ..agents.orchestrator_agent import OrchestratorAgent
from ..core.security import sanitize_input


router = APIRouter(prefix="/api", tags=["agents"])


class ProductIn(BaseModel):
    description: str


@router.post("/classifier/process")
def classifier_process(product: ProductIn) -> Dict[str, Any]:
    agent = CategoryClassifierAgent()
    payload = {"productDisplayName": sanitize_input(product.description), "description": product.description}
    return agent.classify_product(payload)


@router.post("/extractor/process")
def extractor_process(product: ProductIn) -> Dict[str, Any]:
    agent = AttributeExtractorAgent()
    payload = {"productDisplayName": sanitize_input(product.description), "description": product.description}
    return agent.extract_attributes(payload)


@router.post("/tagger/process")
def tagger_process(product: ProductIn) -> Dict[str, Any]:
    agent = TagGeneratorAgent()
    payload = {"productDisplayName": sanitize_input(product.description), "description": product.description}
    return agent.generate_tags(payload)


@router.post("/orchestrator/process")
def orchestrator_process(product: ProductIn) -> Dict[str, Any]:
    agent = OrchestratorAgent()
    payload = {"productDisplayName": sanitize_input(product.description), "description": product.description}
    return agent.process(payload)


