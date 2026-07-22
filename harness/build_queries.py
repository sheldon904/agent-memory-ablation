"""Build the three frozen query sets from the corpus + its ground-truth graph.

Run:  python -m harness.build_queries --seed 1234

The query sets are constructed *before* any arm is run and committed to the
repository (``harness/queries/*.jsonl``). Ground truth is computed by traversing
the corpus's own relation graph, so every answer is machine-checkable:

* known-item (150), a natural question whose answer is one specific fact.
* multi-hop  (30), a question whose answer requires joining 2-3 facts; ground
                     truth is the terminal fact that yields the answer entity
                     plus the full chain.
* distractor (20), a well-formed question about an entity that is NOT in the
                     corpus; the correct behaviour is refusal ("not in memory").

Determinism: one ``random.Random(seed)`` drives all sampling and phrasing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Callable

from corpus.schema import Fact

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QDIR = os.path.join(HERE, "queries")

N_KNOWN = 150
N_MULTI = 30
N_DISTRACT = 20

FIRST_NAMES_EXTRA = ["Jordan", "Blake", "Casey", "Morgan", "Riley", "Skylar",
                     "Quinn", "Alex", "Jamie", "Drew", "Sage", "Rowan"]
LAST_NAMES_EXTRA = ["Ashford", "Bramwell", "Carrow", "Denholm", "Ellery",
                    "Falkner", "Gorsky", "Harlow", "Ivers", "Jessup"]


# ---------------------------------------------------------------------------
# corpus indices
# ---------------------------------------------------------------------------
class Corpus:
    def __init__(self) -> None:
        with open(os.path.join(ROOT, "corpus", "graph.json"), encoding="utf-8") as fh:
            g = json.load(fh)
        self.name: dict[str, str] = {n["id"]: n["name"] for n in g["nodes"]}
        self.type: dict[str, str] = {n["id"]: n["type"] for n in g["nodes"]}
        self.attrs: dict[str, dict] = {n["id"]: n.get("attrs", {}) for n in g["nodes"]}
        self.ids: set[str] = set(self.name)

        # (src, rel) -> [dst]  and  (dst, rel) -> [src]
        self.out: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.inn: dict[tuple[str, str], list[str]] = defaultdict(list)
        for e in g["edges"]:
            self.out[(e["src"], e["rel"])].append(e["dst"])
            self.inn[(e["dst"], e["rel"])].append(e["src"])

        self.facts: list[Fact] = []
        self.fact_by_tri: dict[tuple[str, str, str], str] = {}
        self.facts_by_rel: dict[str, list[Fact]] = defaultdict(list)
        with open(os.path.join(ROOT, "corpus", "facts.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    f = Fact.from_json(json.loads(line))
                    self.facts.append(f)
                    self.fact_by_tri[(f.subject, f.relation, f.object)] = f.fact_id
                    self.facts_by_rel[f.relation].append(f)

        self.people = [i for i, t in self.type.items() if t == "person"]
        self.roles = {i: self.attrs[i].get("role") for i in self.people}

    def one(self, src: str, rel: str) -> str | None:
        vs = self.out.get((src, rel))
        return vs[0] if vs and len(vs) == 1 else (vs[0] if vs else None)

    def uses(self, job: str) -> list[str]:
        return self.out.get((job, "USES"), [])

    def ref(self, eid: str) -> str:
        """A natural-language reference to an entity."""
        t = self.type[eid]
        if t == "job":
            return f"work order {eid}"
        if t == "incident":
            return f"incident {eid}"
        return self.name[eid]


# ---------------------------------------------------------------------------
# known-item templates: relation -> (variants of question builders)
# each builder takes (corpus, fact) -> question string
# ---------------------------------------------------------------------------
def _known_templates() -> dict[str, list[Callable[[Corpus, Fact], str]]]:
    R = lambda c, f: c.ref(f.subject)  # noqa: E731
    return {
        "REPORTS_TO": [lambda c, f: f"Who is {R(c,f)}'s supervisor?",
                       lambda c, f: f"Who does {R(c,f)} report to?"],
        "MEMBER_OF": [lambda c, f: f"Which crew does {R(c,f)} belong to?"],
        "MANAGES": [lambda c, f: f"Which crew does {R(c,f)} lead?"],
        "STATIONED_AT": [lambda c, f: f"Where is {R(c,f)} stationed?",
                         lambda c, f: f"What is {R(c,f)}'s home base?"],
        "OWNED_BY": [lambda c, f: f"Which client owns {R(c,f)}?"],
        "LOCATED_IN": [lambda c, f: f"In which city is {R(c,f)}?"],
        "MANUFACTURED_BY": [lambda c, f: f"Which company manufactured {R(c,f)}?",
                            lambda c, f: f"Who built {R(c,f)}?"],
        "MAINTAINED_BY": [lambda c, f: f"Which vendor services {R(c,f)}?",
                          lambda c, f: f"Who maintains {R(c,f)}?"],
        "HOMED_AT": [lambda c, f: f"Where is {R(c,f)} based?"],
        "HAS_MODEL": [lambda c, f: f"What model is {R(c,f)}?"],
        "ASSIGNED_TO": [lambda c, f: f"Which technician is assigned to {R(c,f)}?",
                        lambda c, f: f"Who is handling {R(c,f)}?"],
        "PERFORMED_AT": [lambda c, f: f"At which site is {R(c,f)} carried out?"],
        "SERVICES": [lambda c, f: f"Which client does {R(c,f)} serve?"],
        "SCHEDULED_ON": [lambda c, f: f"On what date is {R(c,f)} scheduled?"],
        "INVOLVES_EQUIPMENT": [lambda c, f: f"Which piece of equipment was involved in {R(c,f)}?"],
        "INVOLVES_PERSON": [lambda c, f: f"Who was involved in {R(c,f)}?"],
        "OCCURRED_AT": [lambda c, f: f"Where did {R(c,f)} occur?"],
        "REPORTED_ON": [lambda c, f: f"On what date was {R(c,f)} reported?"],
        "HAS_SEVERITY": [lambda c, f: f"What severity level was {R(c,f)} logged at?"],
        "HAS_CERT": [lambda c, f: f"Does {R(c,f)} hold the {c.name[f.object]} certification?"],
    }


def build_known(c: Corpus, rng: random.Random) -> list[dict[str, Any]]:
    templates = _known_templates()
    rels = sorted(templates)
    # round-robin across relations for an even spread
    pools = {r: rng.sample(c.facts_by_rel[r], len(c.facts_by_rel[r])) for r in rels}
    cursor = {r: 0 for r in rels}
    out: list[dict[str, Any]] = []
    ri = 0
    while len(out) < N_KNOWN:
        r = rels[ri % len(rels)]
        ri += 1
        if cursor[r] < len(pools[r]):
            f = pools[r][cursor[r]]
            cursor[r] += 1
            builder = rng.choice(templates[r])
            q = builder(c, f)
            out.append({
                "query_id": f"KI-{len(out)+1:03d}",
                "type": "known_item",
                "text": q,
                "relation": r,
                "answer_fact_ids": [f.fact_id],
                "answer_entities": [f.object] if f.object_is_entity else [],
                "chain_fact_ids": [f.fact_id],
                "hops": 1,
                "should_refuse": False,
            })
        if all(cursor[r] >= len(pools[r]) for r in rels):
            break
    return out


# ---------------------------------------------------------------------------
# multi-hop templates
# ---------------------------------------------------------------------------
def build_multi(c: Corpus, rng: random.Random) -> list[dict[str, Any]]:
    incidents = sorted(i for i, t in c.type.items() if t == "incident")
    jobs = sorted(i for i, t in c.type.items() if t == "job")
    people = sorted(c.people)

    def tri(s: str, r: str, d: str) -> str | None:
        return c.fact_by_tri.get((s, r, d))

    specs: list[dict[str, Any]] = []

    def add(tpl: str, text: str, terminal: str | None, chain: list[str | None],
            answer: str, hops: int) -> None:
        if terminal is None or any(x is None for x in chain) or answer is None:
            return
        specs.append({
            "tpl": tpl, "text": text, "answer_fact_ids": [terminal],
            "chain_fact_ids": [x for x in chain if x], "answer_entities": [answer],
            "hops": hops,
        })

    # --- 2-hop templates ---
    # T1: job -> USES -> equip -> MAINTAINED_BY -> vendor
    for j in jobs:
        eqs = c.uses(j)
        if len(eqs) != 1:
            continue
        e = eqs[0]
        v = c.one(e, "MAINTAINED_BY")
        if not v:
            continue
        add("T1", f"Which vendor maintains the equipment used on work order {j}?",
            tri(e, "MAINTAINED_BY", v), [tri(j, "USES", e), tri(e, "MAINTAINED_BY", v)],
            v, 2)
    # T3: incident -> INVOLVES_EQUIPMENT -> equip -> MANUFACTURED_BY -> vendor
    for i in incidents:
        e = c.one(i, "INVOLVES_EQUIPMENT")
        v = c.one(e, "MANUFACTURED_BY") if e else None
        if e and v:
            add("T3", f"Which company manufactured the equipment involved in incident {i}?",
                tri(e, "MANUFACTURED_BY", v),
                [tri(i, "INVOLVES_EQUIPMENT", e), tri(e, "MANUFACTURED_BY", v)], v, 2)
    # T4: person -> STATIONED_AT -> site -> OWNED_BY -> client
    for p in people:
        s = c.one(p, "STATIONED_AT")
        cl = c.one(s, "OWNED_BY") if s else None
        if s and cl:
            add("T4", f"Which client owns the site where {c.name[p]} is stationed?",
                tri(s, "OWNED_BY", cl),
                [tri(p, "STATIONED_AT", s), tri(s, "OWNED_BY", cl)], cl, 2)
    # T5: incident -> INVOLVES_PERSON -> person -> STATIONED_AT -> site
    for i in incidents:
        p = c.one(i, "INVOLVES_PERSON")
        s = c.one(p, "STATIONED_AT") if p else None
        if p and s:
            add("T5", f"At which site is the person involved in incident {i} stationed?",
                tri(p, "STATIONED_AT", s),
                [tri(i, "INVOLVES_PERSON", p), tri(p, "STATIONED_AT", s)], s, 2)
    # T6: incident -> INVOLVES_EQUIPMENT -> equip -> MAINTAINED_BY -> vendor
    for i in incidents:
        e = c.one(i, "INVOLVES_EQUIPMENT")
        v = c.one(e, "MAINTAINED_BY") if e else None
        if e and v:
            add("T6", f"Which vendor maintains the equipment involved in incident {i}?",
                tri(e, "MAINTAINED_BY", v),
                [tri(i, "INVOLVES_EQUIPMENT", e), tri(e, "MAINTAINED_BY", v)], v, 2)
    # T7: job -> ASSIGNED_TO -> person -> REPORTS_TO -> supervisor (2 hops)
    for j in jobs:
        p = c.one(j, "ASSIGNED_TO")
        if not p or c.roles.get(p) == "director":
            continue
        sup = c.one(p, "REPORTS_TO")
        if sup:
            add("T7", f"Who supervises the technician assigned to work order {j}?",
                tri(p, "REPORTS_TO", sup),
                [tri(j, "ASSIGNED_TO", p), tri(p, "REPORTS_TO", sup)], sup, 2)
    # T8: job -> USES -> equip -> HOMED_AT -> site (2 hops)
    for j in jobs:
        eqs = c.uses(j)
        if len(eqs) != 1:
            continue
        e = eqs[0]
        s = c.one(e, "HOMED_AT")
        if s:
            add("T8", f"At which site is the home base of the equipment used on work order {j}?",
                tri(e, "HOMED_AT", s),
                [tri(j, "USES", e), tri(e, "HOMED_AT", s)], s, 2)
    # T10: job -> ASSIGNED_TO -> person -> STATIONED_AT -> site (2 hops)
    for j in jobs:
        p = c.one(j, "ASSIGNED_TO")
        s = c.one(p, "STATIONED_AT") if p else None
        if p and s:
            add("T10", f"At which site is the technician assigned to work order {j} stationed?",
                tri(p, "STATIONED_AT", s),
                [tri(j, "ASSIGNED_TO", p), tri(p, "STATIONED_AT", s)], s, 2)

    # --- 3-hop templates ---
    # T9: incident -> INVOLVES_EQUIPMENT -> equip -> HOMED_AT -> site -> OWNED_BY -> client
    for i in incidents:
        e = c.one(i, "INVOLVES_EQUIPMENT")
        s = c.one(e, "HOMED_AT") if e else None
        cl = c.one(s, "OWNED_BY") if s else None
        if e and s and cl:
            add("T9", f"Which client owns the home site of the equipment involved in incident {i}?",
                tri(s, "OWNED_BY", cl),
                [tri(i, "INVOLVES_EQUIPMENT", e), tri(e, "HOMED_AT", s), tri(s, "OWNED_BY", cl)],
                cl, 3)
    # T12: job -> USES -> equip -> HOMED_AT -> site -> OWNED_BY -> client
    for j in jobs:
        eqs = c.uses(j)
        if len(eqs) != 1:
            continue
        e = eqs[0]
        s = c.one(e, "HOMED_AT")
        cl = c.one(s, "OWNED_BY") if s else None
        if s and cl:
            add("T12", f"Which client owns the home site of the equipment used on work order {j}?",
                tri(s, "OWNED_BY", cl),
                [tri(j, "USES", e), tri(e, "HOMED_AT", s), tri(s, "OWNED_BY", cl)], cl, 3)
    # T14: incident -> INVOLVES_PERSON -> person -> STATIONED_AT -> site -> OWNED_BY -> client
    for i in incidents:
        p = c.one(i, "INVOLVES_PERSON")
        s = c.one(p, "STATIONED_AT") if p else None
        cl = c.one(s, "OWNED_BY") if s else None
        if p and s and cl:
            add("T14", f"Which client owns the site where the person involved in incident {i} is stationed?",
                tri(s, "OWNED_BY", cl),
                [tri(i, "INVOLVES_PERSON", p), tri(p, "STATIONED_AT", s), tri(s, "OWNED_BY", cl)],
                cl, 3)

    # Round-robin across templates so the 30 queries are spread across templates
    # and hop-counts (the 3-hop templates T9/T12/T14 get equal footing).
    by_tpl: dict[str, list[dict]] = defaultdict(list)
    for s in specs:
        by_tpl[s["tpl"]].append(s)
    keys = sorted(by_tpl)
    for k in keys:
        rng.shuffle(by_tpl[k])
    out: list[dict[str, Any]] = []
    ci = 0
    while len(out) < N_MULTI and any(by_tpl[k] for k in keys):
        k = keys[ci % len(keys)]
        ci += 1
        if by_tpl[k]:
            s = by_tpl[k].pop()
            out.append({
                "query_id": f"MH-{len(out)+1:03d}",
                "type": "multi_hop",
                "text": s["text"],
                "answer_fact_ids": s["answer_fact_ids"],
                "answer_entities": s["answer_entities"],
                "chain_fact_ids": s["chain_fact_ids"],
                "hops": s["hops"],
                "should_refuse": False,
            })
    return out


# ---------------------------------------------------------------------------
# distractors: well-formed questions about entities not in the corpus
# ---------------------------------------------------------------------------
def build_distractors(c: Corpus, rng: random.Random) -> list[dict[str, Any]]:
    existing_names = {c.name[i].lower() for i in c.ids}

    def fake_person() -> str:
        while True:
            nm = f"{rng.choice(FIRST_NAMES_EXTRA)} {rng.choice(LAST_NAMES_EXTRA)}"
            if nm.lower() not in existing_names:
                return nm

    def fake_id(prefix: str, lo: int, hi: int, width: int) -> str:
        while True:
            n = rng.randint(lo, hi)
            eid = f"{prefix}-{n:0{width}d}" if width else f"{prefix}-{n}"
            if eid not in c.ids:
                return eid

    templates: list[Callable[[], str]] = [
        lambda: f"Who is {fake_person()}'s supervisor?",
        lambda: f"Which crew does {fake_person()} belong to?",
        lambda: f"Where is {fake_person()} stationed?",
        lambda: f"Which vendor maintains excavator {fake_id('EX', 700, 999, 0)}?",
        lambda: f"Who manufactured generator {fake_id('GN', 700, 999, 0)}?",
        lambda: f"Which technician is assigned to work order {fake_id('WO', 90000, 99999, 0)}?",
        lambda: f"On what date is work order {fake_id('WO', 90000, 99999, 0)} scheduled?",
        lambda: f"Where did incident {fake_id('INC', 800, 999, 3)} occur?",
        lambda: f"Which client owns {fake_id('SITE', 700, 999, 3)}?",
        lambda: f"What severity level was incident {fake_id('INC', 800, 999, 3)} logged at?",
    ]
    out: list[dict[str, Any]] = []
    ti = 0
    while len(out) < N_DISTRACT:
        q = templates[ti % len(templates)]()
        ti += 1
        out.append({
            "query_id": f"DX-{len(out)+1:03d}",
            "type": "distractor",
            "text": q,
            "answer_fact_ids": [],
            "answer_entities": [],
            "chain_fact_ids": [],
            "hops": 0,
            "should_refuse": True,
        })
    return out


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(seed: int = 1234) -> dict[str, Any]:
    os.makedirs(QDIR, exist_ok=True)
    c = Corpus()
    rng = random.Random(seed)
    known = build_known(c, rng)
    multi = build_multi(c, rng)
    distract = build_distractors(c, rng)

    paths = {
        "known_item": os.path.join(QDIR, "known_item.jsonl"),
        "multi_hop": os.path.join(QDIR, "multi_hop.jsonl"),
        "distractor": os.path.join(QDIR, "distractor.jsonl"),
    }
    _write_jsonl(paths["known_item"], known)
    _write_jsonl(paths["multi_hop"], multi)
    _write_jsonl(paths["distractor"], distract)

    hop_hist: dict[int, int] = defaultdict(int)
    for m in multi:
        hop_hist[m["hops"]] += 1
    manifest = {
        "seed": seed,
        "counts": {"known_item": len(known), "multi_hop": len(multi),
                   "distractor": len(distract)},
        "multi_hop_by_hops": dict(sorted(hop_hist.items())),
        "sha256": {k: _sha256(v) for k, v in paths.items()},
    }
    with open(os.path.join(QDIR, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the frozen query sets.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    print(json.dumps(generate(seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
