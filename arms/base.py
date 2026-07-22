"""The one interface all three memory arms implement.

An arm is handed the frozen corpus once (``build``) and thereafter answers
``recall(query, k)``. Every metric the harness computes is derived from the
``RecallResponse`` an arm returns plus the wall-clock time of the call, so the
arms never see the ground truth and cannot special-case the eval.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

from corpus.schema import Fact


@dataclass
class Recall:
    """One recalled line."""

    fact_id: str | None
    text: str
    score: float
    channel: str
    entities: list[str] = field(default_factory=list)


@dataclass
class RecallResponse:
    """What every arm returns for a query.

    ``results`` is ranked best-first. ``refused`` is the arm's own decision that
    it has no confident answer (the fabrication-pressure signal): on a distractor
    query the correct behaviour is ``refused=True``; on an answerable query it is
    ``refused=False``. ``channels`` carries per-channel diagnostics for tracing.
    """

    query: str
    results: list[Recall]
    refused: bool
    channels: dict[str, int] = field(default_factory=dict)

    def top_entities(self, k: int | None = None) -> list[str]:
        """Distinct entity ids across the (top-k) results, best-rank first."""
        seen: list[str] = []
        rows = self.results if k is None else self.results[:k]
        for r in rows:
            for e in r.entities:
                if e not in seen:
                    seen.append(e)
        return seen

    def fact_ids(self, k: int | None = None) -> list[str]:
        rows = self.results if k is None else self.results[:k]
        return [r.fact_id for r in rows if r.fact_id is not None]


class MemoryProvider(ABC):
    """Abstract base every arm subclasses."""

    #: short, stable arm identifier used in results/traces
    name: str = "base"

    @abstractmethod
    def build(self, facts: list[Fact]) -> None:
        """Index the corpus. Called exactly once."""

    @abstractmethod
    def recall(self, query: str, k: int = 5) -> RecallResponse:
        """Return a ranked recall for the query."""

    @abstractmethod
    def storage_bytes(self) -> int:
        """On-disk bytes this arm's index occupies (for bytes/fact)."""

    def close(self) -> None:  # optional
        pass


def load_corpus(facts_path: str | None = None) -> list[Fact]:
    """Load the frozen corpus as a list of ``Fact``."""
    if facts_path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        facts_path = os.path.join(here, "corpus", "facts.jsonl")
    out: list[Fact] = []
    with open(facts_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Fact.from_json(json.loads(line)))
    return out


def dir_bytes(path: str) -> int:
    """Total bytes of a file or of all files under a directory."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total
