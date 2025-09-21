'''*from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ProductData(BaseModel):
    id: Optional[str] = None
    gender: Optional[str] = None
    masterCategory: Optional[str] = None
    subCategory: Optional[str] = None
    articleType: Optional[str] = None
    baseColour: Optional[str] = None
    season: Optional[str] = None
    year: Optional[int] = None
    usage: Optional[str] = None
    productDisplayName: Optional[str] = None
    filename: Optional[str] = None
    link: Optional[str] = None

class ClassificationResult(BaseModel):
    category: str
    subcategory: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

class AttributeValue(BaseModel):
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class AttributesResult(BaseModel):
    color: AttributeValue
    material: AttributeValue
    size: AttributeValue
    pattern: AttributeValue
    style: AttributeValue
    gender: AttributeValue
    seasonality: AttributeValue
    occasion: AttributeValue
    entities: Optional[List[Dict[str, Any]]] = None

class Tag(BaseModel):
    tag: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Optional[str] = None

class ProcessingResult(BaseModel):
    classification: ClassificationResult
    attributes: AttributesResult
    tags: List[Tag]
    similar_products: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    model: str
    '''

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ProductData(BaseModel):
    description: str

class ClassificationResult(BaseModel):
    category: str
    subcategory: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

class AttributeValue(BaseModel):
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class AttributesResult(BaseModel):
    color: AttributeValue
    material: AttributeValue
    size: AttributeValue
    pattern: AttributeValue
    style: AttributeValue
    gender: AttributeValue
    seasonality: AttributeValue
    occasion: AttributeValue
    entities: Optional[List[Dict[str, Any]]] = None

class Tag(BaseModel):
    tag: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Optional[str] = None

class ProcessingResult(BaseModel):
    classification: ClassificationResult
    attributes: AttributesResult
    tags: List[Tag]
    similar_products: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    model: str