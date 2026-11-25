from typing import List, Sequence, Tuple

from .semantic import SemanticRetriever
from .tfidf import TfidfRetriever


class HybridRetriever:
    """
    Hybrid evidence retrieval using Reciprocal Rank Fusion (RRF).

    Combines lexical TF-IDF and semantic embedding rankings.
    Fused scores represent ranking agreement, not entailment,
    factual correctness, or clinical support.
    """

    def __init__(
        self,
        tfidf_retriever=None,
        semantic_retriever=None,
        top_k: int = 8,
        rrf_k: int = 60,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")

        self.top_k = top_k
        self.rrf_k = rrf_k
        self.tfidf = tfidf_retriever or TfidfRetriever(top_k=top_k)
        self.semantic = semantic_retriever or SemanticRetriever(top_k=top_k)

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

        if any(not passage for passage in documents):
            raise ValueError("passages must not contain empty text")

        rankings = [
            self.tfidf.rank(query, documents),
            self.semantic.rank(query, documents),
        ]

        fused_scores = {}

        for ranking in rankings:
            seen_indices = set()
            for position, (passage_index, _) in enumerate(ranking, start=1):
                if not isinstance(passage_index, int) or isinstance(passage_index, bool):
                    raise ValueError("retriever returned a non-integer passage index")
                if passage_index < 0 or passage_index >= len(documents):
                    raise ValueError("retriever returned an out-of-range passage index")
                if passage_index in seen_indices:
                    raise ValueError("retriever returned a duplicate passage index")
                seen_indices.add(passage_index)
                fused_scores[passage_index] = (
                    fused_scores.get(passage_index, 0.0)
                    + 1.0 / (self.rrf_k + position)
                )

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return [
            (index, float(score))
            for index, score in ranked[:self.top_k]
        ]
