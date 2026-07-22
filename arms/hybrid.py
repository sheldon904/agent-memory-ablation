"""HYBRID arm: a ported slice of the five-channel design, facts + vectors + graph.

Three grounded channels, fused by Reciprocal Rank Fusion (RRF):

* facts, a BM25 lexical index over fact text (the "keyword/FTS" channel);
* vector, MiniLM cosine top-k (shared embedder with the RAG arm);
* graph, entity-linked bounded traversal (shared ``GraphCore`` with the
           graph arm).

RRF is the parameter-free rank fusion the system-under-ablation's own writeup
named as its top recommended improvement, so the hybrid here is a faithful,
minimal realisation of that recommendation rather than a bespoke tuned stack.

Refusal policy (the fabrication-pressure surface): the hybrid refuses only when
*both* grounded-answer channels are silent, the vector top is below the cosine
floor AND the graph links no seed entity. The BM25 channel is a recall booster,
not a grounding signal (a keyword query almost always matches *some* line), so
it never suppresses a refusal on its own. §"Where the hybrid loses" reports the
consequence: this permissive OR-grounding does not recover the graph arm's clean
absence signal.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
from collections import defaultdict

from corpus.schema import Fact
from .base import MemoryProvider, Recall, RecallResponse
from .embeddings import Embedder, get_embedder
from .graph import GraphCore, tokens
from .rag import SIM_FLOOR
from .vector_index import VectorIndex

_WORD_RE = re.compile(r"[a-z0-9]+")


class BM25:
    """Okapi BM25 over fact text, with a persisted on-disk inverted index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: dict[str, int] = {}
        self.tf: dict[str, dict[str, int]] = {}  # term -> {fact_id: tf}
        self.df: dict[str, int] = {}
        self.avgdl = 0.0
        self.N = 0

    def _terms(self, text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    def build(self, facts: list[Fact]) -> None:
        self.doc_ids = [f.fact_id for f in facts]
        total = 0
        for f in facts:
            terms = self._terms(f.text)
            self.doc_len[f.fact_id] = len(terms)
            total += len(terms)
            counts: dict[str, int] = defaultdict(int)
            for t in terms:
                counts[t] += 1
            for t, c in counts.items():
                self.tf.setdefault(t, {})[f.fact_id] = c
        self.N = len(facts)
        self.avgdl = total / self.N if self.N else 0.0
        self.df = {t: len(post) for t, post in self.tf.items()}

    def query(self, query: str, k: int) -> list[tuple[str, float]]:
        qterms = set(self._terms(query))
        scores: dict[str, float] = defaultdict(float)
        for t in qterms:
            post = self.tf.get(t)
            if not post:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for fid, f_td in post.items():
                dl = self.doc_len[fid]
                denom = f_td + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[fid] += idf * (f_td * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def persist(self, db_path: str) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)
        con = sqlite3.connect(db_path)
        try:
            con.execute("CREATE TABLE docs(fact_id TEXT PRIMARY KEY, len INTEGER)")
            con.execute("CREATE TABLE postings(term TEXT, fact_id TEXT, tf INTEGER)")
            con.execute("CREATE INDEX ix_term ON postings(term)")
            with con:
                con.executemany(
                    "INSERT INTO docs VALUES (?,?)", list(self.doc_len.items())
                )
                rows = [
                    (t, fid, c)
                    for t, post in self.tf.items()
                    for fid, c in post.items()
                ]
                con.executemany("INSERT INTO postings VALUES (?,?,?)", rows)
        finally:
            con.close()


# Channel weights for the fusion. The graph channel is the precision-oriented
# one: it returns few, structurally-grounded facts, and is the only channel that
# can reach a multi-hop answer. Equal-weight RRF drowns that lone deep-hop fact
# beneath the two high-recall channels' agreement on shallow, surface-similar
# lines, so the traversal channel is up-weighted. These are fixed, pre-committed
# constants (see the design-constants table in the paper), not per-query tuning.
CHANNEL_WEIGHTS = {"vector": 1.0, "facts": 1.0, "graph": 2.0}


def rrf(rankings: dict[str, list[str]], k0: int = 60,
        weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion across channel orderings."""
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    for channel, order in rankings.items():
        w = weights.get(channel, 1.0)
        for rank, fid in enumerate(order):
            scores[fid] += w / (k0 + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Hybrid(MemoryProvider):
    name = "hybrid"

    def __init__(self, index_dir: str, embedder: Embedder | None = None,
                 max_hops: int = 3, sim_floor: float = SIM_FLOOR,
                 channel_depth: int = 20) -> None:
        self.index_dir = index_dir
        self.embedder = embedder or get_embedder()
        self.max_hops = max_hops
        self.sim_floor = sim_floor
        self.channel_depth = channel_depth
        self._by_id: dict[str, Fact] = {}
        self.core = GraphCore()
        self.bm25 = BM25()
        self._vec = VectorIndex(self.embedder.dim, os.path.join(index_dir, "hybrid_vec.db"))
        self._bm25_db = os.path.join(index_dir, "hybrid_bm25.db")
        self._graph_db = os.path.join(index_dir, "hybrid_graph.db")

    def build(self, facts: list[Fact]) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        self._by_id = {f.fact_id: f for f in facts}
        self.core.index_facts(facts)
        self.core.persist(self._graph_db)
        self.bm25.build(facts)
        self.bm25.persist(self._bm25_db)
        ids = [f.fact_id for f in facts]
        vectors = self.embedder.encode([f.text for f in facts])
        self._vec.build(ids, vectors)

    def recall(self, query: str, k: int = 5) -> RecallResponse:
        depth = max(k, self.channel_depth)

        # channel: vector
        qv = self.embedder.encode([query])[0]
        vec_hits = self._vec.query(qv, depth)
        vec_top = vec_hits[0][1] if vec_hits else 0.0
        vec_order = [fid for fid, _ in vec_hits]

        # channel: facts (BM25)
        bm_hits = self.bm25.query(query, depth)
        bm_order = [fid for fid, _ in bm_hits]

        # channel: graph traversal
        seeds = self.core.linker.link(query)
        graph_recalls = self.core.candidate_facts(query, seeds, self.max_hops, limit=depth)
        graph_order = [r.fact_id for r in graph_recalls if r.fact_id]

        fused = rrf(
            {"vector": vec_order, "facts": bm_order, "graph": graph_order},
            weights=CHANNEL_WEIGHTS,
        )

        results: list[Recall] = []
        for fid, score in fused[:k]:
            f = self._by_id[fid]
            results.append(
                Recall(
                    fact_id=fid,
                    text=f.text,
                    score=round(score, 6),
                    channel="hybrid",
                    entities=list(f.entities),
                )
            )

        # Grounded-answer channels: vector clears the floor, or the graph links.
        grounded = (vec_top >= self.sim_floor) or (len(seeds) > 0)
        refused = not grounded
        return RecallResponse(
            query=query,
            results=results,
            refused=refused,
            channels={
                "vector_hits": len(vec_order),
                "bm25_hits": len(bm_order),
                "graph_seeds": len(seeds),
                "graph_hits": len(graph_order),
            },
        )

    def storage_bytes(self) -> int:
        # The hybrid materialises all three stores: vectors + BM25 + graph.
        # Counting all three is what makes the storage-cost comparison honest.
        total = self._vec.storage_bytes()
        for p in (self._bm25_db, self._graph_db):
            if os.path.exists(p):
                total += os.path.getsize(p)
        return total
