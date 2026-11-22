from typing import List, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """
    Lexical retrieval baseline for evidence passages.

    Similarity scores are not entailment or clinical support.
    """

    def __init__(self, top_k: int = 8, min_score: float = 0.0):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        self.top_k = top_k
        self.min_score = min_score

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

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        try:
            matrix = vectorizer.fit_transform(
                [query.strip()] + documents
            )
        except ValueError as exc:
            if 'empty vocabulary' in str(exc).lower():
                return []
            raise

        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:],
        ).ravel()

        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )

        return [
            (index, float(score))
            for index, score in ranked[:self.top_k]
            if score > self.min_score
        ]
