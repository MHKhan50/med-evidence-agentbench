import pytest

from medevidence_agentbench.retrieval.semantic import SemanticRetriever


@pytest.fixture(scope="module")
def retriever():
    return SemanticRetriever(
        device="mps",
        local_files_only=True,
        top_k=3,
    )


def test_rank_returns_semantically_relevant_passage_first(retriever):
    passages = [
        "The medicine is indicated for seasonal allergies.",
        "Metformin helps regulate blood glucose levels in people with diabetes.",
        "The product should be stored below room temperature.",
    ]

    results = retriever.rank(
        "How does metformin help control blood sugar in diabetic patients?",
        passages,
    )

    assert results
    assert results[0][0] == 1
    assert results[0][1] > 0


def test_rank_returns_original_passage_indices(retriever):
    passages = [
        "Information about product storage conditions.",
        "Metformin helps regulate blood glucose in adults with diabetes.",
    ]

    results = retriever.rank(
        "Blood sugar regulation in people with diabetes",
        passages,
    )

    assert results[0][0] == 1


def test_rank_returns_empty_for_no_passages(retriever):
    assert retriever.rank("medical evidence query", []) == []


def test_rank_rejects_empty_query(retriever):
    with pytest.raises(ValueError):
        retriever.rank("   ", ["A valid evidence passage."])


def test_rank_rejects_empty_passage(retriever):
    with pytest.raises(ValueError):
        retriever.rank("medical evidence", ["valid passage", "   "])


def test_constructor_validates_top_k():
    with pytest.raises(ValueError):
        SemanticRetriever(top_k=0)


def test_constructor_validates_min_score():
    with pytest.raises(ValueError):
        SemanticRetriever(min_score=1.5)


def test_constructor_rejects_empty_model_name():
    with pytest.raises(ValueError):
        SemanticRetriever(model_name="   ")


def test_ranking_is_deterministic(retriever):
    passages = [
        "Metformin helps regulate blood glucose.",
        "Metformin helps regulate blood glucose.",
        "Unrelated product storage information.",
    ]

    first = retriever.rank("Metformin and blood glucose regulation", passages)
    second = retriever.rank("Metformin and blood glucose regulation", passages)

    assert first == second
