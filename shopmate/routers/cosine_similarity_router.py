"""
Cosine Similarity Semantic Router
---------------------------------
Routes user queries (e.g., product vs chitchat) based on embeddings
and cosine similarity with multilingual support.
"""

import os
import json
import numpy as np
from typing import List, Dict
from shopmate.config import EMBEDDINGS

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class CosineSimilarityRouter:
    def __init__(self, lang: str = "en", threshold: float = 0.45):
        """
        Initialize the semantic router.

        Args:
            lang (str): Language code ("en", "ar", etc.)
            threshold (float): Minimum similarity score to accept a route
        """
        self.lang = lang
        self.threshold = threshold
        self.embedding = EMBEDDINGS
        
        self.routes = self._load_routes()
        
        self.route_embeddings = {
            route: [self.embedding.embed_query(prompt) for prompt in prompts]
            for route, prompts in self.routes.items()
        }
        
    def _load_routes(self) -> Dict[str, List[str]]:
        """
        Load route prompts from JSON files inside
        shopmate/routers/templates/locales/{lang}/
        """
        
        base_dir = os.path.join(
            os.path.dirname(__file__), "templates", "locales", self.lang
        )
        
        routes: Dict[str, List[str]] = {}
        
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Locale folder not found: {base_dir}")
        
        for filename in os.listdir(base_dir):
            if filename.endswith(".json"):
                route_name = filename.replace(".json", "")
                file_path = os.path.join(base_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    routes[route_name] = json.load(f)
        
        if not routes:
            raise ValueError(f"No routes found for language: {self.lang}")
        
        return routes
    
    def guide(self, query: str) -> str:
        """
        Route a query to the most likely category.

        Args:
            query (str): user input

        Returns:
            str: route name (e.g., "product", "chitchat", or "unknown")
        """
        
        query_embed = self.embedding.embed_query(query)
        best_route = "unknown"
        best_score = -1.0
        for route, embeddings in self.route_embeddings.items():
            maximum_similarity = max(
                cosine_similarity(query_embed, emb)
                for emb in embeddings
            )
            if maximum_similarity > best_score:
                best_score =  maximum_similarity
                best_route = route
        
        return best_route if best_score >= self.threshold else "unknown"