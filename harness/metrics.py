"""Metric definitions. Pure functions over per-query result records.

A *record* is a plain dict produced by the runner for one (arm, query) pair.
Keeping the metrics pure and separate from the runner is what makes them unit
testable and keeps the definitions auditable in one place.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable

# Rank cutoffs reported for known-item retrieval.
KNOWN_CUTOFFS = (1, 5, 20)
# Depth at which multi-hop success is judged.
MULTIHOP_K = 10


def rank_of(gold_ids: Iterable[str], ranked_ids: list[str]) -> int | None:
    """1-indexed rank of the first gold id in a ranked list, else None."""
    gold = set(gold_ids)
    for i, fid in enumerate(ranked_ids, start=1):
        if fid in gold:
            return i
    return None


def known_item_record(query: dict[str, Any], ranked_ids: list[str],
                      refused: bool, latency_ms: float,
                      top_score: float) -> dict[str, Any]:
    r = rank_of(query["answer_fact_ids"], ranked_ids)
    rec = {
        "query_id": query["query_id"],
        "type": "known_item",
        "relation": query.get("relation"),
        "gold_rank": r,
        "rr": (1.0 / r) if r else 0.0,
        "refused": refused,
        "latency_ms": latency_ms,
        "top_score": top_score,
    }
    for c in KNOWN_CUTOFFS:
        rec[f"hit@{c}"] = 1 if (r is not None and r <= c) else 0
    return rec


def multi_hop_record(query: dict[str, Any], ranked_ids: list[str],
                     result_entities: list[str], refused: bool,
                     latency_ms: float, top_score: float) -> dict[str, Any]:
    top = ranked_ids[:MULTIHOP_K]
    terminal = set(query["answer_fact_ids"])
    chain = set(query["chain_fact_ids"])
    ans_ents = set(query["answer_entities"])
    covered = len(chain & set(top)) / (len(chain) or 1)
    return {
        "query_id": query["query_id"],
        "type": "multi_hop",
        "hops": query["hops"],
        # primary: did the arm surface the terminal (answer-bearing) fact?
        "terminal_hit": 1 if terminal & set(top) else 0,
        # secondary: is the answer entity recoverable from what it returned?
        "entity_recovered": 1 if ans_ents & set(result_entities) else 0,
        "chain_coverage": covered,
        "refused": refused,
        "latency_ms": latency_ms,
        "top_score": top_score,
    }


def distractor_record(query: dict[str, Any], refused: bool,
                      latency_ms: float, top_score: float) -> dict[str, Any]:
    return {
        "query_id": query["query_id"],
        "type": "distractor",
        "refused": refused,
        # correct behaviour on a distractor is to refuse
        "refusal_correct": 1 if refused else 0,
        "latency_ms": latency_ms,
        "top_score": top_score,
    }


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all records for a single arm into the reported metrics."""
    ki = [r for r in records if r["type"] == "known_item"]
    mh = [r for r in records if r["type"] == "multi_hop"]
    dx = [r for r in records if r["type"] == "distractor"]
    lat = [r["latency_ms"] for r in records]

    out: dict[str, Any] = {}
    n_ki = len(ki) or 1
    for c in KNOWN_CUTOFFS:
        out[f"recall@{c}"] = sum(r[f"hit@{c}"] for r in ki) / n_ki
    out["mrr"] = sum(r["rr"] for r in ki) / n_ki
    out["ki_over_refusal"] = sum(1 for r in ki if r["refused"]) / n_ki

    n_mh = len(mh) or 1
    out["multihop_success"] = sum(r["terminal_hit"] for r in mh) / n_mh
    out["multihop_entity_recall"] = sum(r["entity_recovered"] for r in mh) / n_mh
    out["multihop_chain_coverage"] = sum(r["chain_coverage"] for r in mh) / n_mh
    out["mh_over_refusal"] = sum(1 for r in mh if r["refused"]) / n_mh

    n_dx = len(dx) or 1
    out["refusal_correct_distractor"] = sum(r["refusal_correct"] for r in dx) / n_dx

    # over-refusal across ALL answerable queries (known-item + multi-hop)
    answerable = ki + mh
    out["over_refusal_answerable"] = (
        sum(1 for r in answerable if r["refused"]) / (len(answerable) or 1)
    )

    out["latency_p50_ms"] = _p(lat, 0.50)
    out["latency_p95_ms"] = _p(lat, 0.95)
    out["latency_mean_ms"] = statistics.fmean(lat) if lat else 0.0
    out["n_queries"] = len(records)
    return out
