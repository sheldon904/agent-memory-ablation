"""Arm behaviour, built on the hermetic hash embedder for speed/determinism."""

import os

import pytest

from arms.base import load_corpus
from arms.embeddings import HashEmbedder
from arms.graph import GraphOnly
from arms.hybrid import Hybrid
from arms.rag import StatelessRAG


@pytest.fixture(scope="module")
def facts():
    return load_corpus()


@pytest.fixture(scope="module")
def built(tmp_path_factory, facts):
    emb = HashEmbedder()
    d = tmp_path_factory.mktemp("idx")
    arms = {
        "rag": StatelessRAG(str(d / "rag"), embedder=emb),
        "graph": GraphOnly(str(d / "graph")),
        "hybrid": Hybrid(str(d / "hybrid"), embedder=emb),
    }
    for a in arms.values():
        a.build(facts)
    return arms


def test_all_arms_return_ranked_results(built):
    q = "Who is the supervisor of the technician on work order WO-10047?"
    for name, arm in built.items():
        r = arm.recall(q, k=5)
        assert len(r.results) >= 1, name
        scores = [x.score for x in r.results]
        assert scores == sorted(scores, reverse=True), name


def test_storage_bytes_positive(built):
    for name, arm in built.items():
        assert arm.storage_bytes() > 0, name


def test_graph_refuses_on_absent_entity(built):
    r = built["graph"].recall("Who is Nonexistent Person's supervisor?", k=5)
    assert r.refused is True


def test_graph_links_and_answers_known_item(built):
    r = built["graph"].recall("Which client owns SITE-001?", k=5)
    assert r.refused is False
    assert any(x.fact_id for x in r.results)


def test_rag_recall_is_deterministic(built):
    q = "Which vendor maintains the equipment used on work order WO-10047?"
    a = built["rag"].recall(q, k=10).fact_ids()
    b = built["rag"].recall(q, k=10).fact_ids()
    assert a == b


def test_hybrid_channels_reported(built):
    r = built["hybrid"].recall("Who is handling work order WO-10112?", k=5)
    assert "vector_hits" in r.channels
    assert "graph_seeds" in r.channels


def test_refusal_flag_types(built):
    for arm in built.values():
        r = arm.recall("some query about equipment", k=5)
        assert isinstance(r.refused, bool)
