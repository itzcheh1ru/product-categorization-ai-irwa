import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
import os

class InformationRetrieval:
    def __init__(self, data_path: str = "backend/data/cleaned_product_data.csv"):
        self.data_path = data_path
        self.products = self.load_data()
        self.vocabulary_: Dict[str, int] = {}
        self.idf_vector: np.ndarray | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.index_vectors()
    
    def load_data(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.data_path):
            if self.data_path.endswith(".csv"):
                df = pd.read_csv(self.data_path)
                return df.to_dict(orient="records")
            elif self.data_path.endswith(".json"):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        return []
    
    def index_vectors(self):
        if not self.products:
            return

        product_texts = [
            f"{p.get('productDisplayName', '')} {p.get('articleType', '')} {p.get('usage', '')}"
            for p in self.products
        ]

        tokens_list: List[List[str]] = [self._tokenize(text) for text in product_texts]
        self._build_vocabulary(tokens_list)
        term_freq_matrix = self._compute_term_frequency(tokens_list)
        self.idf_vector = self._compute_idf(term_freq_matrix)
        self.tfidf_matrix = self._apply_tfidf(term_freq_matrix, self.idf_vector)
    
    def search_similar_products(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.products:
            return []

        if self.tfidf_matrix is None or self.idf_vector is None:
            return []

        query_tokens = self._tokenize(query)
        query_tf = self._vectorize_tokens(query_tokens)
        query_tfidf = query_tf * self.idf_vector
        similarities = self._cosine_similarity(query_tfidf, self.tfidf_matrix)
        
        # Get top k most similar products
        similar_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in similar_indices:
            if similarities[idx] > 0:
                product = self.products[idx].copy()
                product['similarity_score'] = float(similarities[idx])
                results.append(product)
        
        return results
    
    def get_category_examples(self, category: str, max_examples: int = 3) -> List[str]:
        examples = [
            p['productDisplayName'] for p in self.products 
            if p.get('masterCategory', '').lower() == category.lower()
        ][:max_examples]
        return examples

    def _tokenize(self, text: str) -> List[str]:
        tokens = [
            token for token in ''.join(
                [ch.lower() if ch.isalnum() else ' ' for ch in text]
            ).split()
            if len(token) > 1
        ]
        return tokens

    def _build_vocabulary(self, documents_tokens: List[List[str]]):
        vocab: Dict[str, int] = {}
        for tokens in documents_tokens:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocabulary_ = vocab

    def _compute_term_frequency(self, documents_tokens: List[List[str]]) -> np.ndarray:
        vocab_size = len(self.vocabulary_)
        doc_count = len(documents_tokens)
        tf = np.zeros((doc_count, vocab_size), dtype=np.float64)
        for i, tokens in enumerate(documents_tokens):
            for token in tokens:
                j = self.vocabulary_.get(token)
                if j is not None:
                    tf[i, j] += 1.0
            row_sum = tf[i].sum()
            if row_sum > 0:
                tf[i] /= row_sum
        return tf

    def _compute_idf(self, tf: np.ndarray) -> np.ndarray:
        doc_freq = (tf > 0).sum(axis=0)
        num_docs = tf.shape[0]
        idf = np.log((1 + num_docs) / (1 + doc_freq)) + 1.0
        return idf

    def _apply_tfidf(self, tf: np.ndarray, idf: np.ndarray) -> np.ndarray:
        return tf * idf

    def _vectorize_tokens(self, tokens: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocabulary_), dtype=np.float64)
        for token in tokens:
            j = self.vocabulary_.get(token)
            if j is not None:
                vec[j] += 1.0
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    def _cosine_similarity(self, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vec_norm = np.linalg.norm(vec) + 1e-12
        mat_norms = np.linalg.norm(matrix, axis=1) + 1e-12
        dots = matrix @ vec
        return dots / (mat_norms * vec_norm)
