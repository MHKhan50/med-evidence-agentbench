import pytest

from medevidence_agentbench.retrieval.hybrid import HybridRetriever


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def rank(self, query, passages):
        return self.results


def make_retriever(tfidf_results, semantic_results, **kwargs):
    return HybridRetriever(
        tfidf_retriever=FakeRetriever(tfidf_results),
        semantic_retriever=FakeRetriever(semantic_results),
        **kwargs,
    )


def test_rrf_rewards_passages_ranked_by_both_retrievers():
    retriever = make_retriever(
        [(0, 0.9), (1, 0.7)],
        [(1, 0.95), (0, 0.6)],
        rrf_k=60,
    )

    results = retriever.rank(
        "evidence query",
        ["passage zero", "passage one"],
    )

    assert results[0][0] == 0
    assert results[1][0] == 1
    assert results[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert results[1][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_combines_disagreeing_rankings():
    retriever = make_retriever(
        [(2, 0.9), (0, 0.7)],
        [(1, 0.95), (2, 0.6)],
    )

    results = retriever.rank(
        "evidence query",
        ["passage zero", "passage one", "passage two"],
    )

    assert results[0][0] == 2


def test_rrf_preserves_original_passage_indices():
    retriever = make_retriever(
        [(2, 0.9)],
        [(2, 0.8)],
    )

    results = retriever.rank(
        "evidence query",
        ["first", "second", "third"],
    )

    assert results[0][0] == 2


def test_rrf_returns_deterministic_results():
    retriever = make_retriever(
        [(0, 0.9), (1, 0.8)],
        [(1, 0.9), (0, 0.8)],
    )

    passages = ["first", "second"]

    assert retriever.rank("query", passages) == retriever.rank(
        "query", passages
    )


def test_rrf_returns_empty_for_no_passages():
    retriever = make_retriever([], [])

    assert retriever.rank("evidence query", []) == []


def test_rrf_rejects_empty_query():
    retriever = make_retriever([], [])

    with pytest.raises(ValueError):
        retriever.rank("   ", ["valid passage"])


def test_rrf_rejects_empty_passage():
    retriever = make_retriever([], [])

    with pytest.raises(ValueError):
        retriever.rank("evidence query", ["valid passage", "   "])


def test_constructor_validates_top_k():
    with pytest.raises(ValueError):
        make_retriever([], [], top_k=0)


def test_constructor_validates_rrf_k():
    with pytest.raises(ValueError):
        make_retriever([], [], rrf_k=0)


def test_rrf_respects_top_k():
    retriever = make_retriever(
        [(0, 0.9), (1, 0.8), (2, 0.7)],
        [(0, 0.9), (1, 0.8), (2, 0.7)],
        top_k=2,
    )

    results = retriever.rank(
        "evidence query",
        ["first", "second", "third"],
    )

    assert len(results) == 2


def test_rrf_rejects_negative_passage_index():
    retriever = make_retriever(
        [(-1, 0.9)],
        [],
    )

    with pytest.raises(ValueError, match="out-of-range"):
        retriever.rank("evidence query", ["valid passage"])


def test_rrf_rejects_out_of_range_passage_index():
    retriever = make_retriever(
        [(2, 0.9)],
        [],
    )

    with pytest.raises(ValueError, match="out-of-range"):
        retriever.rank("evidence query", ["first", "second"])


def test_rrf_rejects_duplicate_passage_indices():
    retriever = make_retriever(
        [(0, 0.9), (0, 0.8)],
        [],
    )

    with pytest.raises(ValueError, match="duplicate"):
        retriever.rank("evidence query", ["valid passage"])
