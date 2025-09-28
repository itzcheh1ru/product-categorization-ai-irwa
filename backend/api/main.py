from fastapi import FastAPI, Query, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
from .routes import router as agent_router


app = FastAPI(title="Product Categorization AI API")

# Allow local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cleaned_product_data.csv"


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


class SuggestResponse(BaseModel):
    query: str
    results: List[Suggestion]


class _SearchEngine:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.text_columns = ["productDisplayName", "articleType", "usage", "baseColour", "gender"]

    def _load_df(self) -> pd.DataFrame:
        if self.df is None:
            if not DATA_PATH.exists():
                # Fallback: try product.csv
                alt = DATA_PATH.parent / "product.csv"
                if alt.exists():
                    self.df = pd.read_csv(alt)
                else:
                    self.df = pd.DataFrame()
            else:
                self.df = pd.read_csv(DATA_PATH)
        return self.df

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
            ngram_range=(1, 3),  # Include trigrams for better color matching
            max_features=2000,
            lowercase=True
        )
        self.matrix = self.vectorizer.fit_transform(product_texts)

    def ensure_ready(self):
        if self.vectorizer is None or self.matrix is None:
            self._build_matrix()

    def suggest(self, query: str, top_n: int = 5) -> List[Suggestion]:
        self.ensure_ready()
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []
        
        # Preprocess query for better matching
        processed_query = self._preprocess_query(query)
        query_vec = self.vectorizer.transform([processed_query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        
        # Apply color and type boosting
        sims = self._apply_boosting(query, sims)
        
        top_idx = sims.argsort()[::-1][:top_n]
        df = self._load_df()
        results: List[Suggestion] = []
        for i in top_idx:
            row = df.iloc[i] if not df.empty else {}
            results.append(
                Suggestion(
                    index=int(i),
                    productDisplayName=(row.get("productDisplayName") if hasattr(row, "get") else None),
                    articleType=(row.get("articleType") if hasattr(row, "get") else None),
                    usage=(row.get("usage") if hasattr(row, "get") else None),
                    baseColour=(row.get("baseColour") if hasattr(row, "get") else None),
                    gender=(row.get("gender") if hasattr(row, "get") else None),
                    score=float(sims[i]),
                    filename=(row.get("filename") if hasattr(row, "get") else None),
                    link=(row.get("link") if hasattr(row, "get") else None),
                )
            )
        return results
    
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
                
                # Apply precision boosting logic
                if color_match and type_match:
                    # Both color and type match - HIGHEST priority
                    boost_factor *= 5.0  # 400% boost for exact match
                elif type_match:
                    # Only type match - HIGH priority
                    boost_factor *= 2.5  # 150% boost for type match
                elif color_match:
                    # Only color match - MEDIUM priority
                    boost_factor *= 1.1  # 10% boost for color match
                else:
                    # No match - LOWER priority
                    boost_factor *= 0.1  # Significantly reduce score for no match
                
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
    results = engine.suggest(q, top_n=top_n)
    return {"query": q, "results": results}

# Mount agent routes
app.include_router(agent_router, dependencies=[Depends(verify_api_key)])


