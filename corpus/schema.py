"""Typed schema for the synthetic corpus and its ground-truth relation graph.

The corpus is a fictional industrial field-services company, "Meridian Field
Services." Every fact in the corpus is derived from exactly one graph edge or
node attribute, so every fact has a known subject, relation, object, and set of
linked entities. That is what makes known-item retrieval, multi-hop joins, and
distractor (absence) queries all have machine-checkable ground truth.

The relation vocabulary is *closed* on purpose: it mirrors the closed-vocabulary
knowledge graph in the Hermes hybrid-memory design under ablation here. An
extractor that could mint arbitrary predicates is exactly the assumption this
study freezes and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

# Node (entity) types. `date` and `city` are attribute-like nodes that still get
# first-class graph citizenship so "what happened on <date>" style joins work.
NODE_TYPES: tuple[str, ...] = (
    "person",
    "team",
    "client",
    "site",
    "vendor",
    "equipment",
    "job",
    "incident",
    "cert",
    "city",
    "date",
)

# The closed relation vocabulary. Traversal, multi-hop joins, and the graph
# arm's neighbor listings are all defined over exactly these predicates.
RELATIONS: tuple[str, ...] = (
    # org
    "REPORTS_TO", # person  -> person (supervisor)
    "MEMBER_OF", # person  -> team
    "MANAGES", # person  -> team   (supervisor leads team)
    "HAS_CERT", # person  -> cert
    "STATIONED_AT", # person  -> site   (home base)
    # site / client
    "OWNED_BY", # site    -> client
    "LOCATED_IN", # site    -> city
    "SUPPLIES", # vendor  -> site
    # equipment
    "MANUFACTURED_BY", # equipment -> vendor (manufacturer)
    "MAINTAINED_BY", # equipment -> vendor (service provider)
    "HOMED_AT", # equipment -> site  (where the asset lives)
    "HAS_MODEL", # equipment -> model literal
    # job / work order
    "ASSIGNED_TO", # job -> person
    "USES", # job -> equipment
    "PERFORMED_AT", # job -> site
    "SERVICES", # job -> client
    "SCHEDULED_ON", # job -> date
    # incident
    "INVOLVES_EQUIPMENT", # incident -> equipment
    "INVOLVES_PERSON", # incident -> person
    "OCCURRED_AT", # incident -> site
    "REPORTED_ON", # incident -> date
    "HAS_SEVERITY", # incident -> severity literal
)

RELATIONS_SET = frozenset(RELATIONS)

# Fact categories used for stratified query sampling and composition reporting.
CATEGORIES: tuple[str, ...] = (
    "org",
    "site",
    "equipment",
    "job",
    "incident",
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """A node in the ground-truth graph."""

    id: str
    type: str
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Edge:
    """A directed, typed edge in the ground-truth graph."""

    src: str
    rel: str
    dst: str

    def to_json(self) -> dict[str, Any]:
        return {"src": self.src, "rel": self.rel, "dst": self.dst}


@dataclass(frozen=True)
class Fact:
    """One natural-language statement derived from one edge/attribute.

    `entities` is the set of entity ids the fact mentions (subject and object
    when the object is an entity). It is the shared linking target the graph and
    hybrid arms resolve query mentions against.
    """

    fact_id: str
    text: str
    relation: str
    subject: str
    object: str            # entity id, or a literal (model string, severity)
    object_is_entity: bool
    entities: list[str]
    category: str
    source_ref: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Fact":
        return Fact(
            fact_id=d["fact_id"],
            text=d["text"],
            relation=d["relation"],
            subject=d["subject"],
            object=d["object"],
            object_is_entity=d["object_is_entity"],
            entities=list(d["entities"]),
            category=d["category"],
            source_ref=d["source_ref"],
        )
