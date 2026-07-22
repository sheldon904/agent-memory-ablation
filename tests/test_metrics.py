"""Metric definitions."""

from harness import metrics as M


def test_rank_of():
    assert M.rank_of(["F2"], ["F1", "F2", "F3"]) == 2
    assert M.rank_of(["F9"], ["F1", "F2"]) is None
    assert M.rank_of(["F1", "F3"], ["F5", "F3", "F1"]) == 2


def test_known_item_record_hits_and_rr():
    q = {"query_id": "KI-1", "answer_fact_ids": ["F2"], "relation": "REPORTS_TO"}
    rec = M.known_item_record(q, ["F1", "F2", "F3"], refused=False,
                              latency_ms=1.0, top_score=0.9)
    assert rec["gold_rank"] == 2
    assert rec["rr"] == 0.5
    assert rec["hit@1"] == 0 and rec["hit@5"] == 1 and rec["hit@20"] == 1


def test_multi_hop_terminal_and_entity():
    q = {"query_id": "MH-1", "hops": 2, "answer_fact_ids": ["FT"],
         "chain_fact_ids": ["FA", "FT"], "answer_entities": ["VEN-1"]}
    rec = M.multi_hop_record(q, ["FA", "FT", "FX"], ["VEN-1", "EQ-1"],
                             refused=False, latency_ms=1.0, top_score=0.5)
    assert rec["terminal_hit"] == 1
    assert rec["entity_recovered"] == 1
    assert rec["chain_coverage"] == 1.0

    miss = M.multi_hop_record(q, ["FA", "FZ"], ["EQ-1"], refused=False,
                             latency_ms=1.0, top_score=0.5)
    assert miss["terminal_hit"] == 0
    assert miss["entity_recovered"] == 0
    assert miss["chain_coverage"] == 0.5


def test_distractor_refusal_correct():
    q = {"query_id": "DX-1"}
    assert M.distractor_record(q, True, 1.0, 0.1)["refusal_correct"] == 1
    assert M.distractor_record(q, False, 1.0, 0.5)["refusal_correct"] == 0


def test_aggregate_percentiles_and_rollup():
    records = [
        M.known_item_record({"query_id": "KI-1", "answer_fact_ids": ["F1"],
                             "relation": "R"}, ["F1"], False, 1.0, 0.9),
        M.known_item_record({"query_id": "KI-2", "answer_fact_ids": ["F2"],
                             "relation": "R"}, ["FX", "FY"], True, 3.0, 0.2),
        M.distractor_record({"query_id": "DX-1"}, True, 2.0, 0.1),
    ]
    agg = M.aggregate(records)
    assert agg["recall@1"] == 0.5
    assert agg["mrr"] == 0.5
    assert agg["ki_over_refusal"] == 0.5
    assert agg["refusal_correct_distractor"] == 1.0
    assert agg["latency_p50_ms"] > 0
