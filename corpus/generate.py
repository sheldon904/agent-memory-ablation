"""Deterministic generator for the frozen synthetic corpus.

Run:  python -m corpus.generate --seed 42

Population sizes are fixed constants (seed-independent), so the corpus size is
stable across runs and machines; the seed only controls the random *wiring* of
the graph and the surface phrasing of facts. Output is written to
``corpus/facts.jsonl``, ``corpus/graph.json`` and ``corpus/manifest.json`` and
is committed to the repository as a frozen artifact.

Everything is driven by a single ``random.Random(seed)`` and by deterministic
iteration over sorted lists, so a given seed reproduces byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Any

from .schema import Entity, Edge, Fact, NODE_TYPES, RELATIONS

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Population sizes (fixed; tuned so the corpus lands at ~1,500 facts)
# ---------------------------------------------------------------------------
N_PEOPLE = 60
N_TEAMS = 6
N_CLIENTS = 12
N_SITES = 20
N_VENDORS = 10
N_EQUIPMENT = 45
N_JOBS = 150
N_INCIDENTS = 40
N_CERTS = 6

# ---------------------------------------------------------------------------
# Name banks (deterministic; combined with indices so names are unique)
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Dana", "Marcus", "Priya", "Elena", "Owen", "Tariq", "Nadia", "Colin",
    "Rosa", "Devin", "Ingrid", "Malik", "Yuki", "Grant", "Lena", "Hassan",
    "Bianca", "Tomas", "Aisha", "Reed", "Sonia", "Vince", "Farah", "Cole",
    "Mira", "Desmond", "Petra", "Kwame", "Iris", "Bjorn",
]
LAST_NAMES = [
    "Ruiz", "Feld", "Okafor", "Vance", "Brandt", "Salib", "Kwan", "Ericsson",
    "Delgado", "Hoyt", "Marsh", "Abara", "Tanaka", "Whitlock", "Pham", "Osei",
    "Renner", "Calder", "Nabil", "Stout", "Voss", "Lindqvist", "Amari", "Reyes",
    "Groves", "Mbeki", "Sato", "Halloran", "Frei", "Odom",
]
CITIES = [
    "Tampa", "Lakeland", "Ocala", "Sarasota", "Brandon", "Clearwater",
    "Bartow", "Plant City", "Riverview", "Wesley Chapel",
]
CLIENT_STEMS = [
    "Cedar Ridge", "Harbor Point", "Ironwood", "Silverline", "Northgate",
    "Blue Heron", "Copperfield", "Meadowbrook", "Vantage", "Stonecrest",
    "Kestrel", "Fairwind",
]
CLIENT_SUFFIX = ["Logistics", "Utilities", "Manufacturing", "Foods", "Aggregates",
                 "Properties", "Cold Storage", "Terminals", "Growers", "Materials",
                 "Freight", "Mills"]
SITE_KINDS = ["Depot", "Yard", "Plant", "Terminal", "Facility", "Warehouse",
              "Substation", "Distribution Center"]
VENDOR_STEMS = ["Atlas", "Brenner", "Corvid", "Dynaco", "Everest", "Ferro",
                "Granite", "Helix", "Ironclad", "Juniper"]
VENDOR_SUFFIX = ["Equipment", "Diesel", "Hydraulics", "Power Systems", "Machine Works",
                 "Controls", "Rentals", "Fabrication", "Pumps", "Compressors"]
EQUIP_KINDS = [
    ("excavator", "EX"), ("generator", "GN"), ("compressor", "CP"),
    ("pump", "PM"), ("loader", "LD"), ("chiller", "CH"), ("crane", "CR"),
    ("forklift", "FL"), ("boiler", "BL"), ("HVAC unit", "HV"),
]
MODEL_LETTERS = ["A", "B", "C", "D", "E", "S", "X", "Z"]
CERT_NAMES = [
    "OSHA-30", "Confined Space Entry", "Crane Operator", "Forklift License",
    "HAZMAT Handling", "Arc-Flash Safety",
]
SEVERITIES = ["low", "moderate", "high", "critical"]
INCIDENT_KINDS = [
    "hydraulic leak", "overheating fault", "electrical trip", "seal failure",
    "coolant loss", "bearing wear", "pressure spike", "coupling crack",
]
JOB_KINDS = [
    "preventive maintenance", "emergency repair", "installation",
    "inspection", "overhaul", "calibration", "decommission", "retrofit",
]


def _iso(rng: random.Random) -> str:
    """A deterministic ISO date in the study window (June-July 2026)."""
    month = rng.choice([6, 7])
    day = rng.randint(1, 28)
    return f"2026-{month:02d}-{day:02d}"


class Builder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.entities: dict[str, Entity] = {}
        self.edges: list[Edge] = []
        self.facts: list[Fact] = []
        self._fact_n = 0
        self._date_nodes: dict[str, str] = {}  # iso -> node id

    # -- entity / edge / fact helpers -------------------------------------
    def add_entity(self, eid: str, etype: str, name: str, **attrs: Any) -> str:
        assert etype in NODE_TYPES, etype
        self.entities[eid] = Entity(id=eid, type=etype, name=name, attrs=dict(attrs))
        return eid

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        assert rel in RELATIONS, rel
        self.edges.append(Edge(src=src, rel=rel, dst=dst))

    def date_node(self, iso: str) -> str:
        if iso not in self._date_nodes:
            nid = f"DATE-{iso}"
            self.add_entity(nid, "date", iso)
            self._date_nodes[iso] = nid
        return self._date_nodes[iso]

    def add_fact(
        self,
        text: str,
        relation: str,
        subject: str,
        obj: str,
        object_is_entity: bool,
        category: str,
        source_ref: str,
        extra_entities: tuple[str, ...] = (),
    ) -> Fact:
        self._fact_n += 1
        ents = [subject]
        if object_is_entity:
            ents.append(obj)
        for e in extra_entities:
            if e not in ents:
                ents.append(e)
        f = Fact(
            fact_id=f"F{self._fact_n:05d}",
            text=text,
            relation=relation,
            subject=subject,
            object=obj,
            object_is_entity=object_is_entity,
            entities=ents,
            category=category,
            source_ref=source_ref,
        )
        self.facts.append(f)
        return f

    def name(self, eid: str) -> str:
        return self.entities[eid].name

    # -- population builders ----------------------------------------------
    def build(self) -> None:
        self._people_and_teams()
        self._clients_and_sites()
        self._vendors()
        self._equipment()
        self._jobs()
        self._incidents()

    def _people_and_teams(self) -> None:
        rng = self.rng
        # Certs
        cert_ids = []
        for i in range(N_CERTS):
            cid = f"CERT-{i+1:02d}"
            self.add_entity(cid, "cert", CERT_NAMES[i])
            cert_ids.append(cid)

        # Teams
        team_ids = []
        for i in range(N_TEAMS):
            tid = f"TEAM-{i+1:02d}"
            self.add_entity(tid, "team", f"Crew {chr(ord('A') + i)}")
            team_ids.append(tid)

        # People. Person 0 is the operations director (top of hierarchy).
        # Persons 1..N_TEAMS are the crew supervisors (one per team).
        people_ids = []
        used_names: set[str] = set()
        for i in range(N_PEOPLE):
            # deterministic unique name
            while True:
                fn = FIRST_NAMES[rng.randrange(len(FIRST_NAMES))]
                ln = LAST_NAMES[rng.randrange(len(LAST_NAMES))]
                nm = f"{fn} {ln}"
                if nm not in used_names:
                    used_names.add(nm)
                    break
            pid = f"EMP-{i+1:04d}"
            role = "director" if i == 0 else ("supervisor" if 1 <= i <= N_TEAMS else "technician")
            self.add_entity(pid, "person", nm, role=role)
            people_ids.append(pid)

        director = people_ids[0]
        supervisors = people_ids[1:1 + N_TEAMS]

        # Supervisors manage teams and report to the director.
        for tid, sup in zip(team_ids, supervisors):
            self.add_edge(sup, "MANAGES", tid)
            self.add_fact(
                f"{self.name(sup)} ({sup}) manages {self.name(tid)}.",
                "MANAGES", sup, tid, True, "org", f"org-chart:{sup}",
            )
            self.add_edge(sup, "REPORTS_TO", director)
            self.add_fact(
                f"Supervisor {self.name(sup)} ({sup}) reports to operations "
                f"director {self.name(director)} ({director}).",
                "REPORTS_TO", sup, director, True, "org", f"org-chart:{sup}",
            )

        # Technicians: assign to a team, report to that team's supervisor,
        # station at a site (filled later once sites exist -> store mapping),
        # and hold 1-3 certs.
        self.people_ids = people_ids
        self.team_ids = team_ids
        self.supervisors = supervisors
        self.director = director
        self.cert_ids = cert_ids
        self.team_of: dict[str, str] = {}
        self.sup_of_team: dict[str, str] = dict(zip(team_ids, supervisors))

        technicians = people_ids[1 + N_TEAMS:]
        for pid in technicians:
            tid = rng.choice(team_ids)
            self.team_of[pid] = tid
            self.add_edge(pid, "MEMBER_OF", tid)
            self.add_fact(
                f"Technician {self.name(pid)} ({pid}) is a member of "
                f"{self.name(tid)}.",
                "MEMBER_OF", pid, tid, True, "org", f"roster:{pid}",
            )
            sup = self.sup_of_team[tid]
            self.add_edge(pid, "REPORTS_TO", sup)
            self.add_fact(
                f"{self.name(pid)} ({pid}) reports to {self.name(sup)} ({sup}).",
                "REPORTS_TO", pid, sup, True, "org", f"roster:{pid}",
            )
            n_certs = rng.randint(1, 3)
            for cid in rng.sample(self.cert_ids, n_certs):
                self.add_edge(pid, "HAS_CERT", cid)
                self.add_fact(
                    f"{self.name(pid)} ({pid}) holds the "
                    f"{self.name(cid)} certification.",
                    "HAS_CERT", pid, cid, True, "org", f"training:{pid}",
                )
        # supervisors also belong to their own team + hold certs
        for tid, sup in zip(team_ids, supervisors):
            self.team_of[sup] = tid
            for cid in rng.sample(self.cert_ids, 2):
                self.add_edge(sup, "HAS_CERT", cid)
                self.add_fact(
                    f"{self.name(sup)} ({sup}) holds the {self.name(cid)} "
                    f"certification.",
                    "HAS_CERT", sup, cid, True, "org", f"training:{sup}",
                )

    def _clients_and_sites(self) -> None:
        rng = self.rng
        client_ids = []
        for i in range(N_CLIENTS):
            cid = f"CLI-{i+1:03d}"
            nm = f"{CLIENT_STEMS[i]} {CLIENT_SUFFIX[i % len(CLIENT_SUFFIX)]}"
            self.add_entity(cid, "client", nm)
            client_ids.append(cid)
        self.client_ids = client_ids

        site_ids = []
        for i in range(N_SITES):
            sid = f"SITE-{i+1:03d}"
            city = CITIES[i % len(CITIES)]
            kind = SITE_KINDS[i % len(SITE_KINDS)]
            nm = f"{CLIENT_STEMS[i % len(CLIENT_STEMS)]} {kind}"
            self.add_entity(sid, "site", nm, city=city)
            site_ids.append(sid)
            # OWNED_BY
            owner = client_ids[i % N_CLIENTS]
            self.add_edge(sid, "OWNED_BY", owner)
            self.add_fact(
                f"{self.name(sid)} ({sid}) is owned by {self.name(owner)}.",
                "OWNED_BY", sid, owner, True, "site", f"sites-registry:{sid}",
            )
            # LOCATED_IN (city node)
            city_id = f"CITY-{city.replace(' ', '_')}"
            if city_id not in self.entities:
                self.add_entity(city_id, "city", city)
            self.add_edge(sid, "LOCATED_IN", city_id)
            self.add_fact(
                f"{self.name(sid)} ({sid}) is located in {city}.",
                "LOCATED_IN", sid, city_id, True, "site", f"sites-registry:{sid}",
            )
        self.site_ids = site_ids

        # Station every person at a site now that sites exist.
        for pid in self.people_ids:
            sid = rng.choice(site_ids)
            self.add_edge(pid, "STATIONED_AT", sid)
            self.add_fact(
                f"{self.name(pid)} ({pid}) is stationed at {self.name(sid)} "
                f"({sid}).",
                "STATIONED_AT", pid, sid, True, "org", f"roster:{pid}",
            )

    def _vendors(self) -> None:
        rng = self.rng
        vendor_ids = []
        for i in range(N_VENDORS):
            vid = f"VEN-{i+1:03d}"
            nm = f"{VENDOR_STEMS[i]} {VENDOR_SUFFIX[i % len(VENDOR_SUFFIX)]}"
            self.add_entity(vid, "vendor", nm)
            vendor_ids.append(vid)
        self.vendor_ids = vendor_ids
        # SUPPLIES: each vendor supplies 2-4 sites.
        for vid in vendor_ids:
            for sid in rng.sample(self.site_ids, rng.randint(2, 4)):
                self.add_edge(vid, "SUPPLIES", sid)
                self.add_fact(
                    f"{self.name(vid)} supplies parts and service to "
                    f"{self.name(sid)} ({sid}).",
                    "SUPPLIES", vid, sid, True, "site", f"vendor-contracts:{vid}",
                )

    def _equipment(self) -> None:
        rng = self.rng
        equip_ids = []
        for i in range(N_EQUIPMENT):
            kind, prefix = EQUIP_KINDS[i % len(EQUIP_KINDS)]
            eid = f"{prefix}-{100 + i}"
            nm = f"{kind} {eid}"
            self.add_entity(eid, "equipment", nm, kind=kind)
            equip_ids.append(eid)
            # HAS_MODEL (literal)
            model = f"{MODEL_LETTERS[rng.randrange(len(MODEL_LETTERS))]}{rng.randint(100, 999)}"
            self.add_fact(
                f"{nm} is a model {model} unit.",
                "HAS_MODEL", eid, model, False, "equipment", f"asset-registry:{eid}",
            )
            # MANUFACTURED_BY
            mfr = rng.choice(self.vendor_ids)
            self.add_edge(eid, "MANUFACTURED_BY", mfr)
            self.add_fact(
                f"{nm} was manufactured by {self.name(mfr)}.",
                "MANUFACTURED_BY", eid, mfr, True, "equipment", f"asset-registry:{eid}",
            )
            # MAINTAINED_BY (a different vendor when possible)
            svc_choices = [v for v in self.vendor_ids if v != mfr]
            svc = rng.choice(svc_choices)
            self.add_edge(eid, "MAINTAINED_BY", svc)
            self.add_fact(
                f"{nm} is maintained by {self.name(svc)}.",
                "MAINTAINED_BY", eid, svc, True, "equipment", f"asset-registry:{eid}",
            )
            # HOMED_AT
            home = rng.choice(self.site_ids)
            self.add_edge(eid, "HOMED_AT", home)
            self.add_fact(
                f"{nm} is based at {self.name(home)} ({home}).",
                "HOMED_AT", eid, home, True, "equipment", f"asset-registry:{eid}",
            )
        self.equip_ids = equip_ids

    def _jobs(self) -> None:
        rng = self.rng
        techs = self.people_ids  # anyone can be assigned
        job_ids = []
        for i in range(N_JOBS):
            jid = f"WO-{10000 + i}"
            kind = rng.choice(JOB_KINDS)
            self.add_entity(jid, "job", f"work order {jid}", kind=kind)
            job_ids.append(jid)
            # ASSIGNED_TO
            who = rng.choice(techs)
            self.add_edge(jid, "ASSIGNED_TO", who)
            self.add_fact(
                f"Work order {jid} ({kind}) is assigned to {self.name(who)} "
                f"({who}).",
                "ASSIGNED_TO", jid, who, True, "job", f"dispatch:{jid}",
            )
            # PERFORMED_AT
            site = rng.choice(self.site_ids)
            self.add_edge(jid, "PERFORMED_AT", site)
            self.add_fact(
                f"Work order {jid} is performed at {self.name(site)} ({site}).",
                "PERFORMED_AT", jid, site, True, "job", f"dispatch:{jid}",
            )
            # SERVICES (client owning the site, for a coherent graph)
            owner = self._owner_of_site(site)
            self.add_edge(jid, "SERVICES", owner)
            self.add_fact(
                f"Work order {jid} services the client {self.name(owner)}.",
                "SERVICES", jid, owner, True, "job", f"dispatch:{jid}",
            )
            # USES 1-2 equipment
            for eq in rng.sample(self.equip_ids, rng.randint(1, 2)):
                self.add_edge(jid, "USES", eq)
                self.add_fact(
                    f"Work order {jid} uses {self.name(eq)}.",
                    "USES", jid, eq, True, "job", f"dispatch:{jid}",
                )
            # SCHEDULED_ON
            iso = _iso(rng)
            dn = self.date_node(iso)
            self.add_edge(jid, "SCHEDULED_ON", dn)
            self.add_fact(
                f"Work order {jid} is scheduled on {iso}.",
                "SCHEDULED_ON", jid, dn, True, "job", f"dispatch:{jid}",
            )
        self.job_ids = job_ids

    def _incidents(self) -> None:
        rng = self.rng
        inc_ids = []
        for i in range(N_INCIDENTS):
            iid = f"INC-{i+1:03d}"
            kind = rng.choice(INCIDENT_KINDS)
            self.add_entity(iid, "incident", f"incident {iid}", kind=kind)
            inc_ids.append(iid)
            eq = rng.choice(self.equip_ids)
            self.add_edge(iid, "INVOLVES_EQUIPMENT", eq)
            self.add_fact(
                f"Incident {iid} ({kind}) involves {self.name(eq)}.",
                "INVOLVES_EQUIPMENT", iid, eq, True, "incident", f"safety-log:{iid}",
            )
            person = rng.choice(self.people_ids)
            self.add_edge(iid, "INVOLVES_PERSON", person)
            self.add_fact(
                f"Incident {iid} involves {self.name(person)} ({person}).",
                "INVOLVES_PERSON", iid, person, True, "incident", f"safety-log:{iid}",
            )
            site = rng.choice(self.site_ids)
            self.add_edge(iid, "OCCURRED_AT", site)
            self.add_fact(
                f"Incident {iid} occurred at {self.name(site)} ({site}).",
                "OCCURRED_AT", iid, site, True, "incident", f"safety-log:{iid}",
            )
            iso = _iso(rng)
            dn = self.date_node(iso)
            self.add_edge(iid, "REPORTED_ON", dn)
            self.add_fact(
                f"Incident {iid} was reported on {iso}.",
                "REPORTED_ON", iid, dn, True, "incident", f"safety-log:{iid}",
            )
            sev = rng.choice(SEVERITIES)
            self.add_fact(
                f"Incident {iid} was logged at {sev} severity.",
                "HAS_SEVERITY", iid, sev, False, "incident", f"safety-log:{iid}",
            )
        self.incident_ids = inc_ids

    def _owner_of_site(self, sid: str) -> str:
        for e in self.edges:
            if e.src == sid and e.rel == "OWNED_BY":
                return e.dst
        raise KeyError(sid)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(seed: int = 42, out_dir: str | None = None) -> dict[str, Any]:
    out_dir = out_dir or HERE
    b = Builder(seed)
    b.build()

    facts_path = os.path.join(out_dir, "facts.jsonl")
    graph_path = os.path.join(out_dir, "graph.json")
    manifest_path = os.path.join(out_dir, "manifest.json")

    # facts.jsonl, one fact per line, stable order
    with open(facts_path, "w", encoding="utf-8", newline="\n") as fh:
        for f in b.facts:
            fh.write(json.dumps(f.to_json(), ensure_ascii=False, sort_keys=True) + "\n")

    # graph.json, nodes + edges
    graph = {
        "nodes": [b.entities[k].to_json() for k in sorted(b.entities)],
        "edges": [e.to_json() for e in b.edges],
    }
    with open(graph_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(graph, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    # composition
    cat_counts: dict[str, int] = {}
    rel_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for f in b.facts:
        cat_counts[f.category] = cat_counts.get(f.category, 0) + 1
        rel_counts[f.relation] = rel_counts.get(f.relation, 0) + 1
    for e in b.entities.values():
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    manifest = {
        "seed": seed,
        "n_facts": len(b.facts),
        "n_entities": len(b.entities),
        "n_edges": len(b.edges),
        "facts_by_category": dict(sorted(cat_counts.items())),
        "facts_by_relation": dict(sorted(rel_counts.items())),
        "entities_by_type": dict(sorted(type_counts.items())),
        "populations": {
            "people": N_PEOPLE, "teams": N_TEAMS, "clients": N_CLIENTS,
            "sites": N_SITES, "vendors": N_VENDORS, "equipment": N_EQUIPMENT,
            "jobs": N_JOBS, "incidents": N_INCIDENTS, "certs": N_CERTS,
        },
        "facts_sha256": _sha256(facts_path),
        "graph_sha256": _sha256(graph_path),
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the frozen synthetic corpus.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    manifest = generate(seed=args.seed, out_dir=args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
