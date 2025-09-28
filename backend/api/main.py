from fastapi import FastAPI, Query, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
from core.mongodb_connection import get_mongodb_manager, connect_to_mongodb
# from .routes import router as agent_router


app = FastAPI(title="Product Categorization AI API")

# Allow local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cleaned_product_data.csv"  # Removed: Using MongoDB now


class Suggestion(BaseModel):
    index: int
    productDisplayName: Optional[str] = None
    articleType: Optional[str] = None
    usage: Optional[str] = None
    baseColour: Optional[str] = None
    gender: Optional[str] = None
    score: float
    filename: Optional[str] = None
    link: Optional[str] = None
    match_type: Optional[str] = None  # "exact", "partial", "related"


class SuggestResponse(BaseModel):
    query: str
    exact_matches: List[Suggestion]
    other_recommendations: List[Suggestion]
    total_results: int


class _SearchEngine:
    def __init__(self):
        self.mongodb_manager = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.text_columns = ["productDisplayName", "articleType", "usage", "baseColour", "gender"]
        self._initialized = False
        self._connect_to_mongodb()

    def _connect_to_mongodb(self):
        """Connect to MongoDB and initialize"""
        try:
            if connect_to_mongodb():
                self.mongodb_manager = get_mongodb_manager()
                print("✅ Connected to MongoDB successfully")
            else:
                print("❌ Failed to connect to MongoDB")
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")

    def _load_df(self) -> pd.DataFrame:
        """Load data from MongoDB instead of CSV"""
        if self.mongodb_manager is None:
            return pd.DataFrame()
        
        try:
            # Get products from MongoDB
            products = self.mongodb_manager.get_all_products()
            if not products:
                print("No products found in MongoDB")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(products)
            
            # Select only essential columns
            essential_columns = ['productDisplayName', 'articleType', 'baseColour', 
                               'usage', 'gender', 'filename', 'link']
            available_columns = [col for col in essential_columns if col in df.columns]
            
            if available_columns:
                df = df[available_columns]
            
            print(f"✅ Loaded {len(df)} products from MongoDB")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data from MongoDB: {e}")
            return pd.DataFrame()

    def _build_matrix(self):
        df = self._load_df()
        if df.empty:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
            self.matrix = self.vectorizer.fit_transform([""])
            return
        available = [c for c in self.text_columns if c in df.columns]
        if not available:
            available = [df.select_dtypes(include=["object"]).columns[0]]
        product_texts = df[available[0]].fillna("")
        for col in available[1:]:
            product_texts = product_texts + " " + df[col].fillna("")
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),  # Reduce to bigrams for faster processing
                max_features=1000,   # Reduce features for faster processing
                lowercase=True
            )
        self.matrix = self.vectorizer.fit_transform(product_texts)

    def ensure_ready(self):
        if self.vectorizer is None or self.matrix is None:
            self._build_matrix()

    def suggest_mongodb(self, query: str, top_n: int = 5) -> Dict[str, List[Suggestion]]:
        """Search using MongoDB's native text search capabilities"""
        if self.mongodb_manager is None:
            return {"exact_matches": [], "other_recommendations": []}
        
        try:
            # Use MongoDB's text search
            products = self.mongodb_manager.search_products(query, limit=top_n * 2)
            
            exact_matches: List[Suggestion] = []
            other_recommendations: List[Suggestion] = []
            
            for i, product in enumerate(products):
                suggestion = Suggestion(
                    index=i,
                    productDisplayName=product.get('productDisplayName', ''),
                    articleType=product.get('articleType', ''),
                    usage=product.get('usage', ''),
                    baseColour=product.get('baseColour', ''),
                    gender=product.get('gender', ''),
                    score=0.9 - (i * 0.1),  # Decreasing score
                    filename=product.get('filename', ''),
                    link=product.get('link', ''),
                    match_type="exact" if i < top_n else "related"
                )
                
                if i < top_n:
                    exact_matches.append(suggestion)
                else:
                    other_recommendations.append(suggestion)
            
            return {"exact_matches": exact_matches, "other_recommendations": other_recommendations}
            
        except Exception as e:
            print(f"❌ MongoDB search error: {e}")
            return {"exact_matches": [], "other_recommendations": []}

    def suggest(self, query: str, top_n: int = 5) -> Dict[str, List[Suggestion]]:
        # Try MongoDB search first
        if self.mongodb_manager is not None:
            mongodb_results = self.suggest_mongodb(query, top_n)
            if mongodb_results["exact_matches"] or mongodb_results["other_recommendations"]:
                return mongodb_results
        
        # Fallback to original TF-IDF search
        self.ensure_ready()
        if self.matrix is None or self.matrix.shape[0] == 0:
            return {"exact_matches": [], "other_recommendations": []}

        # Preprocess query for better matching
        processed_query = self._preprocess_query(query)
        query_vec = self.vectorizer.transform([processed_query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        
        # Apply color and type boosting
        sims = self._apply_boosting(query, sims)
        
        # Get fewer results for faster processing
        top_idx = sims.argsort()[::-1][:top_n + 3]  # Get just a few extra results
        df = self._load_df()
        
        # Extract colors and product types for categorization
        colors = self._extract_colors(query)
        product_types = self._extract_product_types(query)
        
        exact_matches: List[Suggestion] = []
        other_recommendations: List[Suggestion] = []
        
        for i in top_idx:
            row = df.iloc[i] if not df.empty else {}
            
            # Check if this is an exact match
            is_exact_match = self._is_exact_match(row, colors, product_types)
            
            suggestion = Suggestion(
                index=int(i),
                productDisplayName=(row.get("productDisplayName") if hasattr(row, "get") else None),
                articleType=(row.get("articleType") if hasattr(row, "get") else None),
                usage=(row.get("usage") if hasattr(row, "get") else None),
                baseColour=(row.get("baseColour") if hasattr(row, "get") else None),
                gender=(row.get("gender") if hasattr(row, "get") else None),
                score=float(sims[i]),
                filename=(row.get("filename") if hasattr(row, "get") else None),
                link=(row.get("link") if hasattr(row, "get") else None),
                match_type="exact" if is_exact_match else "related"
            )
            
            if is_exact_match and len(exact_matches) < top_n:
                exact_matches.append(suggestion)
            elif not is_exact_match and len(other_recommendations) < top_n:
                other_recommendations.append(suggestion)
        
        return {
            "exact_matches": exact_matches,
            "other_recommendations": other_recommendations
        }
    
    def _is_exact_match(self, row: pd.Series, colors: List[str], product_types: List[str]) -> bool:
        """Check if a product is an exact match based on color and product type."""
        if not colors and not product_types:
            return False
            
        product_text = ' '.join([
            str(row.get("productDisplayName", "")),
            str(row.get("articleType", "")),
            str(row.get("baseColour", "")),
            str(row.get("usage", ""))
        ]).lower()
        
        color_match = False
        type_match = False
        
        # Check color match
        if colors:
            for color in colors:
                if color in product_text:
                    color_match = True
                    break
        
        # Check product type match
        if product_types:
            for ptype in product_types:
                if ptype in product_text:
                    type_match = True
                    break
        
        # Exact match requires both color and type to match (if both are specified)
        if colors and product_types:
            return color_match and type_match
        elif colors:
            return color_match
        elif product_types:
            return type_match
        
        return False
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess search query for better matching."""
        import re
        
        # Convert to lowercase
        query = query.lower()
        
        # Expand common abbreviations
        expansions = {
            'jean': 'jeans',
            'pant': 'pants',
            'shirt': 'shirts',
            'dress': 'dresses',
            'shoe': 'shoes',
            'bag': 'bags'
        }
        
        for short, long in expansions.items():
            query = re.sub(r'\b' + short + r'\b', long, query)
        
        # Add color variations
        color_variations = {
            'blue': 'blue navy royal sky',
            'red': 'red crimson scarlet',
            'green': 'green emerald forest',
            'black': 'black dark charcoal',
            'white': 'white cream ivory',
            'brown': 'brown tan beige',
            'gray': 'gray grey silver',
            'pink': 'pink rose magenta',
            'purple': 'purple violet lavender',
            'yellow': 'yellow gold amber'
        }
        
        for color, variations in color_variations.items():
            if color in query:
                query += ' ' + variations
        
        return query
    
    def _apply_boosting(self, query: str, similarities: np.ndarray) -> np.ndarray:
        """Apply boosting for color and product type matches with precision."""
        import re
        
        boosted_similarities = similarities.copy()
        df = self._load_df()
        
        if df.empty:
            return boosted_similarities
        
        try:
            # Extract colors and product types from query
            colors = self._extract_colors(query)
            product_types = self._extract_product_types(query)
            
            # Apply boosting with precision logic
            for i in range(len(similarities)):
                if i >= len(df):
                    break
                    
                row = df.iloc[i]
                boost_factor = 1.0
                color_match = False
                type_match = False
                
                # Check for color match
                if colors:
                    product_text = ' '.join([
                        str(row.get("productDisplayName", "")),
                        str(row.get("articleType", "")),
                        str(row.get("baseColour", "")),
                        str(row.get("usage", ""))
                    ]).lower()
                    
                    for color in colors:
                        if color in product_text:
                            color_match = True
                            break
                
                # Check for product type match
                if product_types:
                    product_text = ' '.join([
                        str(row.get("productDisplayName", "")),
                        str(row.get("articleType", "")),
                        str(row.get("usage", ""))
                    ]).lower()
                    
                    for ptype in product_types:
                        if ptype in product_text:
                            type_match = True
                            break
                
                # Simplified boosting for faster processing
                if color_match and type_match:
                    boost_factor *= 3.0  # Exact match
                elif type_match:
                    boost_factor *= 2.0  # Type match
                elif color_match:
                    boost_factor *= 1.2  # Color match
                else:
                    boost_factor *= 0.5  # No match
                
                boosted_similarities[i] *= boost_factor
                
        except Exception as e:
            print(f"Error applying boosting: {str(e)}")
        
        return boosted_similarities
    
    def _extract_colors(self, query: str) -> list:
        """Extract color terms from query."""
        colors = ['red', 'blue', 'green', 'black', 'white', 'brown', 'gray', 'grey', 
                 'pink', 'purple', 'yellow', 'orange', 'navy', 'royal', 'sky', 'crimson',
                 'scarlet', 'emerald', 'forest', 'dark', 'charcoal', 'cream', 'ivory',
                 'tan', 'beige', 'silver', 'rose', 'magenta', 'violet', 'lavender',
                 'gold', 'amber']
        
        found_colors = []
        query_lower = query.lower()
        for color in colors:
            if color in query_lower:
                found_colors.append(color)
        
        return found_colors
    
    def _extract_product_types(self, query: str) -> list:
        """Extract product type terms from query with synonyms."""
        # Define product types with synonyms
        type_synonyms = {
            'jean': ['jeans', 'jean', 'denim'],
            'pant': ['pants', 'pant', 'trousers', 'trouser'],
            'shirt': ['shirts', 'shirt', 'tshirt', 't-shirt', 'tee'],
            'dress': ['dresses', 'dress', 'frock', 'frocks', 'gown', 'gowns'],
            'shoe': ['shoes', 'shoe', 'footwear', 'sneakers', 'boots'],
            'bag': ['bags', 'bag', 'handbag', 'purse', 'backpack'],
            'jacket': ['jackets', 'jacket', 'coat', 'coats', 'blazer'],
            'sweater': ['sweaters', 'sweater', 'pullover', 'jumper'],
            'hoodie': ['hoodies', 'hoodie', 'hooded'],
            'blouse': ['blouses', 'blouse', 'top', 'tops'],
            'skirt': ['skirts', 'skirt'],
            'short': ['shorts', 'short'],
            'watch': ['watches', 'watch', 'timepiece'],
            'saree': ['sarees', 'saree', 'sari', 'saris'],
            'kurta': ['kurtas', 'kurta', 'kurti', 'kurtis'],
            'legging': ['leggings', 'legging', 'tights'],
            'jeggings': ['jeggings', 'jegging']
        }
        
        found_types = []
        query_lower = query.lower()
        
        # Check for each product type and its synonyms
        for main_type, synonyms in type_synonyms.items():
            for synonym in synonyms:
                if synonym in query_lower:
                    # Add the main type to found_types for consistent matching
                    found_types.append(main_type)
                    break  # Only add once per main type
        
        return found_types


engine = _SearchEngine()


def verify_api_key(x_api_key: str | None = Header(default=None)):
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/search/suggest", response_model=SuggestResponse)
def suggest_products(q: str = Query(..., min_length=1), top_n: int = 5):
    categorized_results = engine.suggest(q, top_n=top_n)
    exact_matches = categorized_results["exact_matches"]
    other_recommendations = categorized_results["other_recommendations"]
    total_results = len(exact_matches) + len(other_recommendations)
    
    return SuggestResponse(
        query=q, 
        exact_matches=exact_matches,
        other_recommendations=other_recommendations,
        total_results=total_results
    )

# Mount agent routes - disabled for performance
# app.include_router(agent_router, dependencies=[Depends(verify_api_key)])

# ==================== RESPONSIBLE AI ENDPOINTS ====================

from core.security import (
    responsible_ai_manager,
    enhanced_sanitize_input_with_ai_safety,
    detect_and_mitigate_bias,
    assess_data_privacy,
    generate_ethical_ai_guidelines,
    validate_ai_model_ethics,
    create_responsible_ai_policy,
    log_responsible_ai_event,
    get_responsible_ai_dashboard_data,
    BiasType,
    PrivacyLevel,
    AISafetyLevel
)

class BiasDetectionRequest(BaseModel):
    text: str
    context: Optional[str] = ""

class PrivacyAssessmentRequest(BaseModel):
    data: Dict[str, Any]
    user_id: Optional[str] = None

class AISafetyRequest(BaseModel):
    content: str
    context: Optional[str] = ""

class ModelEthicsRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    predictions: List[Dict[str, Any]]
    ground_truth: List[Dict[str, Any]]

@app.post("/api/responsible-ai/detect-bias")
def detect_bias_endpoint(request: BiasDetectionRequest):
    """Detect bias in text content"""
    try:
        bias_results = responsible_ai_manager.detect_bias(request.text, request.context)
        
        # Log the bias detection event
        log_responsible_ai_event(
            "bias_detection_request",
            {
                "text_length": len(request.text),
                "bias_count": len(bias_results),
                "bias_types": [result.bias_type.value for result in bias_results]
            }
        )
        
        return {
            "text": request.text,
            "bias_detected": len(bias_results) > 0,
            "bias_results": [
                {
                    "bias_type": result.bias_type.value,
                    "confidence": result.confidence,
                    "severity": result.severity,
                    "description": result.description,
                    "mitigation_suggestion": result.mitigation_suggestion,
                    "detected_patterns": result.detected_patterns
                }
                for result in bias_results
            ]
        }
    except Exception as e:
        logger.error(f"Bias detection error: {e}")
        return {"error": "Bias detection failed", "details": str(e)}

@app.post("/api/responsible-ai/assess-privacy")
def assess_privacy_endpoint(request: PrivacyAssessmentRequest):
    """Assess privacy implications of data processing"""
    try:
        processed_data, privacy_log = assess_data_privacy(request.data, request.user_id)
        
        # Log the privacy assessment event
        log_responsible_ai_event(
            "privacy_assessment",
            {
                "data_keys": list(request.data.keys()),
                "pii_detected": privacy_log.pii_detected,
                "anonymization_applied": privacy_log.anonymization_applied,
                "privacy_level": privacy_log.privacy_level.value
            },
            request.user_id
        )
        
        return {
            "original_data": request.data,
            "processed_data": processed_data,
            "privacy_assessment": {
                "privacy_level": privacy_log.privacy_level.value,
                "pii_detected": privacy_log.pii_detected,
                "anonymization_applied": privacy_log.anonymization_applied,
                "retention_period_days": privacy_log.retention_period,
                "timestamp": privacy_log.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Privacy assessment error: {e}")
        return {"error": "Privacy assessment failed", "details": str(e)}

@app.post("/api/responsible-ai/assess-safety")
def assess_ai_safety_endpoint(request: AISafetyRequest):
    """Assess AI safety of content"""
    try:
        safety_result = responsible_ai_manager.assess_ai_safety(request.content, request.context)
        
        # Log the safety assessment event
        log_responsible_ai_event(
            "safety_assessment",
            {
                "content_length": len(request.content),
                "safety_level": safety_result.safety_level.value,
                "risk_factors_count": len(safety_result.risk_factors),
                "content_flags": safety_result.content_flags
            }
        )
        
        return {
            "content": request.content,
            "safety_assessment": {
                "safety_level": safety_result.safety_level.value,
                "confidence": safety_result.confidence,
                "risk_factors": safety_result.risk_factors,
                "mitigation_actions": safety_result.mitigation_actions,
                "content_flags": safety_result.content_flags
            }
        }
    except Exception as e:
        logger.error(f"AI safety assessment error: {e}")
        return {"error": "AI safety assessment failed", "details": str(e)}

@app.post("/api/responsible-ai/validate-model-ethics")
def validate_model_ethics_endpoint(request: ModelEthicsRequest):
    """Validate AI model ethics and fairness"""
    try:
        ethics_report = validate_ai_model_ethics(
            request.model_name,
            request.predictions,
            request.ground_truth
        )
        
        # Log the model ethics validation event
        log_responsible_ai_event(
            "model_ethics_validation",
            {
                "model_name": request.model_name,
                "predictions_count": len(request.predictions),
                "fairness_score": ethics_report["fairness_metrics"]["overall_fairness_score"],
                "compliance_status": ethics_report["ethical_compliance"]["compliance_status"]
            }
        )
        
        return ethics_report
    except Exception as e:
        logger.error(f"Model ethics validation error: {e}")
        return {"error": "Model ethics validation failed", "details": str(e)}

@app.get("/api/responsible-ai/guidelines")
def get_ethical_guidelines():
    """Get ethical AI guidelines and best practices"""
    try:
        guidelines = generate_ethical_ai_guidelines()
        return guidelines
    except Exception as e:
        logger.error(f"Guidelines retrieval error: {e}")
        return {"error": "Guidelines retrieval failed", "details": str(e)}

@app.get("/api/responsible-ai/policy")
def get_responsible_ai_policy():
    """Get Responsible AI policy document"""
    try:
        policy = create_responsible_ai_policy()
        return policy
    except Exception as e:
        logger.error(f"Policy retrieval error: {e}")
        return {"error": "Policy retrieval failed", "details": str(e)}

@app.get("/api/responsible-ai/dashboard")
def get_responsible_ai_dashboard():
    """Get Responsible AI dashboard data"""
    try:
        dashboard_data = get_responsible_ai_dashboard_data()
        return dashboard_data
    except Exception as e:
        logger.error(f"Dashboard data retrieval error: {e}")
        return {"error": "Dashboard data retrieval failed", "details": str(e)}

@app.get("/api/responsible-ai/report")
def get_responsible_ai_report():
    """Get comprehensive Responsible AI report"""
    try:
        report = responsible_ai_manager.generate_responsible_ai_report()
        return report
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return {"error": "Report generation failed", "details": str(e)}

# ==================== MONGODB ENDPOINTS ====================

@app.get("/api/mongodb/status")
def get_mongodb_status():
    """Get MongoDB connection status and database statistics"""
    try:
        from core.mongodb_connection import get_database_connection_status
        status = get_database_connection_status()
        return {
            "status": "success",
            "mongodb_status": status,
            "message": "MongoDB connection status retrieved successfully"
        }
    except Exception as e:
        logger.error(f"MongoDB status error: {e}")
        return {"error": "MongoDB status retrieval failed", "details": str(e)}

@app.get("/api/mongodb/stats")
def get_mongodb_stats():
    """Get MongoDB database statistics"""
    try:
        mongodb_manager = get_mongodb_manager()
        if mongodb_manager.client is None:
            if not mongodb_manager.connect():
                return {"error": "Failed to connect to MongoDB"}
        
        stats = mongodb_manager.get_database_stats()
        return {
            "status": "success",
            "database_stats": stats,
            "message": "MongoDB statistics retrieved successfully"
        }
    except Exception as e:
        logger.error(f"MongoDB stats error: {e}")
        return {"error": "MongoDB statistics retrieval failed", "details": str(e)}

@app.get("/api/mongodb/search")
def search_mongodb_products(q: str = Query(..., description="Search query"), limit: int = Query(10, description="Number of results")):
    """Search products directly from MongoDB"""
    try:
        mongodb_manager = get_mongodb_manager()
        if mongodb_manager.client is None:
            if not mongodb_manager.connect():
                return {"error": "Failed to connect to MongoDB"}
        
        products = mongodb_manager.search_products(q, limit)
        return {
            "status": "success",
            "query": q,
            "results_count": len(products),
            "products": products[:limit],
            "message": f"Found {len(products)} products matching '{q}'"
        }
    except Exception as e:
        logger.error(f"MongoDB search error: {e}")
        return {"error": "MongoDB search failed", "details": str(e)}

@app.post("/api/responsible-ai/sanitize")
def sanitize_input_endpoint(request: BiasDetectionRequest):
    """Enhanced input sanitization with AI safety assessment"""
    try:
        sanitized, safety_result = enhanced_sanitize_input_with_ai_safety(request.text, request.context)
        
        # Log the sanitization event
        log_responsible_ai_event(
            "input_sanitization",
            {
                "original_length": len(request.text),
                "sanitized_length": len(sanitized),
                "safety_level": safety_result.safety_level.value,
                "content_blocked": safety_result.safety_level == AISafetyLevel.BLOCKED
            }
        )
        
        return {
            "original_input": request.text,
            "sanitized_input": sanitized,
            "safety_assessment": {
                "safety_level": safety_result.safety_level.value,
                "confidence": safety_result.confidence,
                "risk_factors": safety_result.risk_factors,
                "mitigation_actions": safety_result.mitigation_actions,
                "content_flags": safety_result.content_flags
            },
            "content_blocked": safety_result.safety_level == AISafetyLevel.BLOCKED
        }
    except Exception as e:
        logger.error(f"Input sanitization error: {e}")
        return {"error": "Input sanitization failed", "details": str(e)}


