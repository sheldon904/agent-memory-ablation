"""Shared entity linker: resolve entity mentions in a query to entity ids.

This is the closed-vocabulary linking primitive the graph and hybrid arms use to
find their traversal seeds. It is deliberately *not* available to the stateless
RAG arm, which sees only the query string, that asymmetry is the architecture
under test, not an unfair advantage.

Linking is purely lexical and deterministic: explicit id tokens (``EMP-0142``,
``WO-10432``, ``INC-014``, ``EX-114`` ...) via regex, plus whole display-name
substring matches. A query that mentions no corpus entity links to nothing,
which is exactly what makes the graph arm refuse cleanly on distractors.
"""

from __future__ import annotations

import re
from corpus.schema import Entity

# Entity-id shapes that appear verbatim in fact text and queries.
_ID_RE = re.compile(
    r"\b("
    r"EMP-\d{3,4}"          # person
    r"|WO-\d{4,5}"          # job / work order
    r"|INC-\d{2,3}"         # incident
    r"|SITE-\d{2,3}"        # site
    r"|CLI-\d{2,3}"         # client
    r"|VEN-\d{2,3}"         # vendor
    r"|TEAM-\d{1,2}"        # team
    r"|CERT-\d{1,2}"        # cert
    r"|[A-Z]{2}-1\d{2}"     # equipment (EX-100.., GN-1.., etc.)
    r")\b"
)
_DATE_RE = re.compile(r"\b(2026-\d{2}-\d{2})\b")


class EntityLinker:
    def __init__(self, entities: list[Entity]) -> None:
        self._ids: set[str] = {e.id for e in entities}
        # name (lowercased) -> id, compiled with word boundaries so short names
        # like "Crew D" do not spuriously match inside "crew does".
        self._by_name: dict[str, str] = {}
        self._name_re: dict[str, "re.Pattern[str]"] = {}
        for e in entities:
            nm = e.name.lower()
            self._by_name[nm] = e.id
            if len(nm) >= 4:
                self._name_re[nm] = re.compile(r"\b" + re.escape(nm) + r"\b")
        # longest names first so a more specific name claims the mention
        self._names_sorted = sorted(self._name_re, key=len, reverse=True)

    def link(self, query: str) -> list[str]:
        """Return the distinct entity ids mentioned in the query, in order."""
        found: list[str] = []

        def _add(eid: str) -> None:
            if eid in self._ids and eid not in found:
                found.append(eid)

        for m in _ID_RE.finditer(query):
            _add(m.group(1))
        for m in _DATE_RE.finditer(query):
            _add(f"DATE-{m.group(1)}")

        q = query.lower()
        for nm in self._names_sorted:
            if self._name_re[nm].search(q):
                _add(self._by_name[nm])
        return found
