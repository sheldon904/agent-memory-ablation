"""Corpus integrity + determinism."""

import json
import os
import tempfile

from corpus.generate import generate
from corpus.schema import RELATIONS_SET, NODE_TYPES
from arms.base import load_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_graph():
    with open(os.path.join(ROOT, "corpus", "graph.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_fact_count_in_target_band():
    facts = load_corpus()
    assert 1400 <= len(facts) <= 1700, len(facts)


def test_relations_are_closed_vocabulary():
    facts = load_corpus()
    for f in facts:
        assert f.relation in RELATIONS_SET, f.relation


def test_node_types_are_closed():
    g = _load_graph()
    for n in g["nodes"]:
        assert n["type"] in NODE_TYPES, n["type"]


def test_edges_reference_valid_nodes():
    g = _load_graph()
    ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["src"] in ids and e["dst"] in ids, e
        assert e["rel"] in RELATIONS_SET, e["rel"]


def test_fact_entities_exist_in_graph():
    g = _load_graph()
    ids = {n["id"] for n in g["nodes"]}
    facts = load_corpus()
    for f in facts:
        for e in f.entities:
            assert e in ids, (f.fact_id, e)
        if f.object_is_entity:
            assert f.object in ids


def test_generation_is_deterministic():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        m1 = generate(seed=42, out_dir=d1)
        m2 = generate(seed=42, out_dir=d2)
        assert m1["facts_sha256"] == m2["facts_sha256"]
        assert m1["graph_sha256"] == m2["graph_sha256"]


def test_committed_corpus_matches_manifest():
    with open(os.path.join(ROOT, "corpus", "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    with tempfile.TemporaryDirectory() as d:
        fresh = generate(seed=manifest["seed"], out_dir=d)
    assert fresh["facts_sha256"] == manifest["facts_sha256"], "committed corpus is stale"
    assert fresh["graph_sha256"] == manifest["graph_sha256"], "committed graph is stale"
