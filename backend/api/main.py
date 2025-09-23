from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
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
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform([""])
            return
        available = [c for c in self.text_columns if c in df.columns]
        if not available:
            available = [df.select_dtypes(include=["object"]).columns[0]]
        product_texts = df[available[0]].fillna("")
        for col in available[1:]:
            product_texts = product_texts + " " + df[col].fillna("")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(product_texts)

    def ensure_ready(self):
        if self.vectorizer is None or self.matrix is None:
            self._build_matrix()

    def suggest(self, query: str, top_n: int = 5) -> List[Suggestion]:
        self.ensure_ready()
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
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
                )
            )
        return results


engine = _SearchEngine()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/search/suggest", response_model=SuggestResponse)
def suggest_products(q: str = Query(..., min_length=1), top_n: int = 5):
    results = engine.suggest(q, top_n=top_n)
    return {"query": q, "results": results}

# Mount agent routes
app.include_router(agent_router)


