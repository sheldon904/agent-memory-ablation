"""STATELESS RAG arm: chunked text + MiniLM embeddings in sqlite-vec, top-k only.

Each corpus fact is one retrieval chunk (the facts are already sentence-sized).
Recall is pure vector top-k over the MiniLM index with a cosine floor; there is
no graph, no entity linking, no fusion, and no state carried between queries, 
the deliberately minimal baseline the study ablates against.

The cosine floor doubles as the refusal rule: if the best match is below
``sim_floor`` the arm reports "not in memory". That single threshold is the
whole of RAG's defence against fabrication, which is exactly what the distractor
set measures.
"""

from __future__ import annotations

import os

import numpy as np

from corpus.schema import Fact
from .base import MemoryProvider, Recall, RecallResponse
from .embeddings import Embedder, get_embedder
from .vector_index import VectorIndex

# Matches the 0.40 semantic-recall threshold in the system under ablation.
SIM_FLOOR = 0.40


class StatelessRAG(MemoryProvider):
    name = "rag"

    def __init__(self, index_dir: str, embedder: Embedder | None = None,
                 sim_floor: float = SIM_FLOOR) -> None:
        self.index_dir = index_dir
        self.embedder = embedder or get_embedder()
        self.sim_floor = sim_floor
        self._by_id: dict[str, Fact] = {}
        self._index = VectorIndex(self.embedder.dim, os.path.join(index_dir, "rag_vec.db"))

    def build(self, facts: list[Fact]) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        self._by_id = {f.fact_id: f for f in facts}
        ids = [f.fact_id for f in facts]
        vectors = self.embedder.encode([f.text for f in facts])
        self._index.build(ids, vectors)

    def recall(self, query: str, k: int = 5) -> RecallResponse:
        qv = self.embedder.encode([query])[0]
        hits = self._index.query(qv, k)
        results: list[Recall] = []
        for fid, sim in hits:
            f = self._by_id[fid]
            results.append(
                Recall(
                    fact_id=fid,
                    text=f.text,
                    score=round(sim, 6),
                    channel="vector",
                    entities=list(f.entities),
                )
            )
        top = results[0].score if results else 0.0
        refused = top < self.sim_floor
        return RecallResponse(
            query=query,
            results=results,
            refused=refused,
            channels={"vector_hits": len(results)},
        )

    def storage_bytes(self) -> int:
        return self._index.storage_bytes()
