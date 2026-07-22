"""GRAPH-ONLY arm: canonicalized knowledge graph, traversal-based recall.

No vectors. The corpus's ground-truth relation graph is loaded into an
adjacency structure; a query is linked to seed entities and answered by bounded
breadth-first traversal, ranking the facts in the seed neighbourhood by hop
distance and lexical (token-Jaccard) overlap. This is the closed-vocabulary,
graph-first posture of the system under ablation, stripped of every other
channel.

``GraphCore`` is the reusable traversal engine; the hybrid arm imports it so the
two arms share one graph implementation.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3

from corpus.schema import Entity, Fact
from .base import MemoryProvider, Recall, RecallResponse
from .entities import EntityLinker

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of to is are was were on at in by for and or with which who whom "
    "whose what where when how does do did that this these those it its as be "
    "been being from into".split()
)


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class GraphCore:
    """Adjacency + fact indices over the ground-truth graph."""

    def __init__(self, graph_path: str | None = None) -> None:
        if graph_path is None:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            graph_path = os.path.join(here, "corpus", "graph.json")
        with open(graph_path, "r", encoding="utf-8") as fh:
            g = json.load(fh)
        self.entities: list[Entity] = [
            Entity(id=n["id"], type=n["type"], name=n["name"], attrs=n.get("attrs", {}))
            for n in g["nodes"]
        ]
        self._ent_by_id = {e.id: e for e in self.entities}
        self.edges: list[tuple[str, str, str]] = [
            (e["src"], e["rel"], e["dst"]) for e in g["edges"]
        ]
        # FORWARD (src -> dst) adjacency. Relations point outward from an entity
        # to its resources (job -> equipment -> vendor; site -> owner), so every
        # multi-hop chain in the query set is forward-directed. Traversing only
        # forward is what avoids reverse fan-in explosion through hub nodes
        # (a site is the object of hundreds of "work order performed at" facts):
        # the same hub-suppression instinct the source system encodes, expressed
        # here as edge directionality.
        self.fwd: dict[str, list[str]] = {e.id: [] for e in self.entities}
        for s, _rel, d in self.edges:
            if s in self.fwd and d in self.fwd:
                self.fwd[s].append(d)
        for k in self.fwd:
            self.fwd[k] = sorted(set(self.fwd[k]))

        self.linker = EntityLinker(self.entities)
        self.facts_by_entity: dict[str, list[Fact]] = {}
        self.facts_by_subject: dict[str, list[Fact]] = {}
        self._fact_tokens: dict[str, set[str]] = {}

    def index_facts(self, facts: list[Fact]) -> None:
        for f in facts:
            self._fact_tokens[f.fact_id] = tokens(f.text)
            # forward facts: keyed by the subject, so a reached node contributes
            # only its outgoing assertions (never the hub's incoming fan-in).
            self.facts_by_subject.setdefault(f.subject, []).append(f)
            for e in f.entities:
                self.facts_by_entity.setdefault(e, []).append(f)

    def persist(self, db_path: str) -> None:
        """Write nodes + directed edges to sqlite for a real byte measurement."""
        if os.path.exists(db_path):
            os.remove(db_path)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        con = sqlite3.connect(db_path)
        try:
            con.execute("CREATE TABLE nodes(id TEXT PRIMARY KEY, type TEXT, name TEXT)")
            con.execute("CREATE TABLE edges(src TEXT, rel TEXT, dst TEXT)")
            con.execute("CREATE INDEX ix_src ON edges(src)")
            con.execute("CREATE INDEX ix_dst ON edges(dst)")
            with con:
                con.executemany(
                    "INSERT INTO nodes VALUES (?,?,?)",
                    [(e.id, e.type, e.name) for e in self.entities],
                )
                con.executemany("INSERT INTO edges VALUES (?,?,?)", self.edges)
        finally:
            con.close()

    # -- traversal --------------------------------------------------------
    def reachable(self, seeds: list[str], max_hops: int, node_cap: int = 600) -> dict[str, int]:
        """Forward BFS: entity id -> minimum hop distance from any seed."""
        dist: dict[str, int] = {s: 0 for s in seeds if s in self.fwd}
        frontier = list(dist)
        hop = 0
        while frontier and hop < max_hops and len(dist) < node_cap:
            hop += 1
            nxt: list[str] = []
            for node in frontier:
                for nb in self.fwd.get(node, ()):
                    if nb not in dist:
                        dist[nb] = hop
                        nxt.append(nb)
                        if len(dist) >= node_cap:
                            break
                if len(dist) >= node_cap:
                    break
            frontier = nxt
        return dist

    def candidate_facts(
        self, query: str, seeds: list[str], max_hops: int, limit: int
    ) -> list[Recall]:
        if not seeds:
            return []
        dist = self.reachable(seeds, max_hops)
        qtok = tokens(query)
        scored: dict[str, tuple[int, float, Fact]] = {}
        # A reached node contributes only its FORWARD facts (subject == node),
        # so the candidate set is the small BFS-tree of outgoing assertions that
        # every forward-directed chain terminates inside.
        for eid, hop in dist.items():
            for f in self.facts_by_subject.get(eid, ()):
                jac = jaccard(qtok, self._fact_tokens[f.fact_id])
                prev = scored.get(f.fact_id)
                if prev is None or (hop, -jac) < (prev[0], -prev[1]):
                    scored[f.fact_id] = (hop, jac, f)
        ranked = sorted(
            scored.values(), key=lambda t: (t[0], -t[1], t[2].fact_id)
        )
        out: list[Recall] = []
        for fhop, jac, f in ranked[:limit]:
            # graph-native score: closer + more lexically on-point ranks higher
            score = (1.0 / (1 + fhop)) * (0.5 + 0.5 * jac)
            out.append(
                Recall(
                    fact_id=f.fact_id,
                    text=f.text,
                    score=round(score, 6),
                    channel="graph",
                    entities=list(f.entities),
                )
            )
        return out


class GraphOnly(MemoryProvider):
    name = "graph"

    def __init__(self, index_dir: str, max_hops: int = 3) -> None:
        self.index_dir = index_dir
        self.max_hops = max_hops
        self.core = GraphCore()
        self._db_path = os.path.join(index_dir, "graph.db")

    def build(self, facts: list[Fact]) -> None:
        self.core.index_facts(facts)
        os.makedirs(self.index_dir, exist_ok=True)
        # Persist nodes + edges to sqlite so storage_bytes is a real on-disk
        # measurement comparable to the other arms.
        self.core.persist(self._db_path)

    def recall(self, query: str, k: int = 5) -> RecallResponse:
        seeds = self.core.linker.link(query)
        results = self.core.candidate_facts(query, seeds, self.max_hops, limit=k)
        # Graph refuses cleanly when the query names no entity in the store:
        # absence from the canonical graph is an unambiguous "not in memory".
        refused = len(seeds) == 0
        return RecallResponse(
            query=query,
            results=results,
            refused=refused,
            channels={"graph_seeds": len(seeds), "graph_hits": len(results)},
        )

    def storage_bytes(self) -> int:
        return os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0
