"""Query-set integrity: counts, ground-truth validity, determinism."""

import json
import os

from harness.build_queries import Corpus, generate
from arms.base import load_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDIR = os.path.join(ROOT, "harness", "queries")


def _load(name):
    rows = []
    with open(os.path.join(QDIR, f"{name}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_counts():
    assert len(_load("known_item")) == 150
    assert len(_load("multi_hop")) == 30
    assert len(_load("distractor")) == 20


def test_known_item_gold_facts_exist():
    ids = {f.fact_id for f in load_corpus()}
    for q in _load("known_item"):
        assert len(q["answer_fact_ids"]) == 1
        assert q["answer_fact_ids"][0] in ids
        assert q["should_refuse"] is False


def test_multi_hop_chain_and_terminal_valid():
    facts = {f.fact_id: f for f in load_corpus()}
    for q in _load("multi_hop"):
        assert q["hops"] in (2, 3)
        assert q["answer_fact_ids"][0] in facts          # terminal exists
        for cf in q["chain_fact_ids"]:
            assert cf in facts                            # chain facts exist
        # the terminal fact actually yields the answer entity
        term = facts[q["answer_fact_ids"][0]]
        assert q["answer_entities"][0] in term.entities


def test_multi_hop_answer_is_forward_reachable():
    """The answer entity must be reachable from a query-mentioned seed by
    forward traversal, i.e. the graph arm can in principle find it."""
    c = Corpus()
    from arms.graph import GraphCore
    core = GraphCore()
    for q in _load("multi_hop"):
        seeds = core.linker.link(q["text"])
        assert seeds, q["query_id"]
        reach = core.reachable(seeds, max_hops=3)
        assert q["answer_entities"][0] in reach, (q["query_id"], q["text"])


def test_distractors_reference_absent_entities():
    c = Corpus()
    from arms.graph import GraphCore
    core = GraphCore()
    for q in _load("distractor"):
        assert q["should_refuse"] is True
        # a well-formed distractor links to NO real corpus entity
        assert core.linker.link(q["text"]) == [], (q["query_id"], q["text"])


def test_query_generation_deterministic(tmp_path, monkeypatch):
    # generate() writes to the repo QDIR; capture hashes and compare to committed
    with open(os.path.join(QDIR, "manifest.json"), encoding="utf-8") as fh:
        committed = json.load(fh)
    fresh = generate(seed=committed["seed"])
    assert fresh["sha256"] == committed["sha256"], "committed query sets are stale"
