from typing import List, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """
    Semantic evidence retrieval using a SentenceTransformer encoder.

    Similarity scores indicate embedding proximity, not entailment,
    factual correctness, or clinical support.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 8,
        min_score: float = 0.0,
        device: str = None,
        local_files_only: bool = True,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty")

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.top_k = top_k
        self.min_score = min_score
        self.device = device
        self.model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_files_only,
        )

    def rank(
        self,
        query: str,
        passages: Sequence[str],
    ) -> List[Tuple[int, float]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if not passages:
            return []

        documents = [str(p).strip() for p in passages]

        if any(not p for p in documents):
            raise ValueError("passages must not contain empty text")

        embeddings = self.model.encode(
            [query.strip()] + documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        query_embedding = embeddings[0]
        passage_embeddings = embeddings[1:]

        scores = passage_embeddings @ query_embedding

        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )

        return [
            (index, float(score))
            for index, score in ranked[:self.top_k]
            if score > self.min_score
        ]
