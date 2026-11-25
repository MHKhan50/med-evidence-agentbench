import pytest

from medevidence_agentbench.retrieval.tfidf import TfidfRetriever


def test_rank_returns_relevant_passage_first():
    retriever = TfidfRetriever(top_k=3)

    passages = [
        "The medicine is indicated for seasonal allergies.",
        "Metformin improves glycemic control in adults with diabetes.",
        "The product should be stored below room temperature.",
    ]

    results = retriever.rank(
        "Metformin improves glucose control in adults with diabetes.",
        passages,
    )

    assert results
    assert results[0][0] == 1
    assert results[0][1] > 0


def test_rank_returns_original_passage_indices():
    retriever = TfidfRetriever(top_k=2)

    passages = [
        "Unrelated information about storage conditions.",
        "Metformin is used to improve glycemic control.",
    ]

    results = retriever.rank("Metformin glycemic control", passages)

    assert results[0][0] == 1


def test_rank_returns_empty_for_no_passages():
    retriever = TfidfRetriever()

    assert retriever.rank("medical claim", []) == []


def test_rank_rejects_empty_query():
    retriever = TfidfRetriever()

    with pytest.raises(ValueError):
        retriever.rank("   ", ["Some evidence passage."])


def test_rank_rejects_empty_passage():
    retriever = TfidfRetriever()

    with pytest.raises(ValueError):
        retriever.rank("medical claim", ["valid passage", "   "])


def test_constructor_validates_top_k():
    with pytest.raises(ValueError):
        TfidfRetriever(top_k=0)


def test_constructor_validates_min_score():
    with pytest.raises(ValueError):
        TfidfRetriever(min_score=1.5)


def test_ranking_is_deterministic():
    retriever = TfidfRetriever(top_k=3)

    passages = [
        "Metformin improves glycemic control.",
        "Metformin improves glycemic control.",
        "Unrelated storage information.",
    ]

    first = retriever.rank("Metformin glycemic control", passages)
    second = retriever.rank("Metformin glycemic control", passages)

    assert first == second


def test_rank_handles_query_with_only_stop_words():
    retriever = TfidfRetriever()

    results = retriever.rank(
        "the and is",
        ["The medicine is effective."],
    )

    assert results == []


def test_rank_handles_empty_vocabulary():
    retriever = TfidfRetriever()

    results = retriever.rank(
        "the and is",
        ["the and is", "of to in"],
    )

    assert results == []
