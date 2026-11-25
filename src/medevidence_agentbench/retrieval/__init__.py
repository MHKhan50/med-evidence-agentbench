"""Evidence retrieval methods for MedEvidence-AgentBench."""

from importlib import import_module

__all__ = ["TfidfRetriever", "SemanticRetriever", "HybridRetriever"]


def __getattr__(name):
    if name == "TfidfRetriever":
        return import_module(".tfidf", __name__).TfidfRetriever

    if name == "SemanticRetriever":
        return import_module(".semantic", __name__).SemanticRetriever

    if name == "HybridRetriever":
        return import_module(".hybrid", __name__).HybridRetriever

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
