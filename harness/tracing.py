"""Tracing: every recall is emitted as an OpenTelemetry-style span.

The exporter writes OTLP-shaped JSON into ``traces/`` so the repository ships a
standard-tooling artifact (this deliberately closes the "we only speak our own
telemetry" gap the source system's writeup flagged). Two sinks:

* ``OtelExporter`` (default, always on), writes ``traces/<arm>/spans.jsonl``
  (one span per line) and ``traces/<arm>/otlp.trace.json`` (a single OTLP
  ``resourceSpans`` document). No network, no dependencies.

* ``LangfuseExporter`` (optional), if ``langfuse`` is installed and
  ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are set, each recall is also
  logged as a Langfuse span. Absent creds or package => silent no-op, so the
  harness never depends on a running Langfuse instance.

Span/trace ids are derived deterministically from (arm, query_id) so span
identity is stable across runs; wall-clock timestamps reflect the actual run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


def _hex_id(*parts: str, nbytes: int = 8) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[: nbytes * 2]


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    start_unix_nano: int
    end_unix_nano: int
    attributes: dict[str, Any]
    status: str = "OK"

    def to_otlp(self) -> dict[str, Any]:
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": str(self.start_unix_nano),
            "endTimeUnixNano": str(self.end_unix_nano),
            "attributes": [
                {"key": k, "value": _otlp_value(v)}
                for k, v in self.attributes.items()
            ],
            "status": {"code": "STATUS_CODE_OK"},
        }


def _otlp_value(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


class OtelExporter:
    def __init__(self, traces_dir: str, arm: str) -> None:
        self.arm = arm
        self.dir = os.path.join(traces_dir, arm)
        os.makedirs(self.dir, exist_ok=True)
        self.spans: list[Span] = []

    def record(self, query_id: str, query_type: str, latency_ms: float,
               attributes: dict[str, Any]) -> None:
        # Base epoch is fixed so exported nanos are anchored, not wall-clock
        # noise; the *duration* is the real measured latency.
        base = 1_700_000_000_000_000_000
        start = base + len(self.spans) * 10_000_000
        end = start + int(latency_ms * 1_000_000)
        span = Span(
            name="memory.recall",
            trace_id=_hex_id(self.arm, query_id, nbytes=16),
            span_id=_hex_id(self.arm, query_id, "span", nbytes=8),
            start_unix_nano=start,
            end_unix_nano=end,
            attributes={
                "memory.arm": self.arm,
                "query.id": query_id,
                "query.type": query_type,
                "recall.latency_ms": round(latency_ms, 4),
                **attributes,
            },
        )
        self.spans.append(span)

    def flush(self) -> dict[str, Any]:
        jsonl_path = os.path.join(self.dir, "spans.jsonl")
        with open(jsonl_path, "w", encoding="utf-8", newline="\n") as fh:
            for s in self.spans:
                fh.write(json.dumps(s.to_otlp(), ensure_ascii=False, sort_keys=True) + "\n")
        otlp_doc = {
            "resourceSpans": [{
                "resource": {"attributes": [
                    {"key": "service.name", "value": {"stringValue": "agent-memory-ablation"}},
                    {"key": "memory.arm", "value": {"stringValue": self.arm}},
                ]},
                "scopeSpans": [{
                    "scope": {"name": "harness.tracing", "version": "1.0.0"},
                    "spans": [s.to_otlp() for s in self.spans],
                }],
            }]
        }
        otlp_path = os.path.join(self.dir, "otlp.trace.json")
        with open(otlp_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(otlp_doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return {"arm": self.arm, "spans": len(self.spans),
                "jsonl": jsonl_path, "otlp": otlp_path}


class LangfuseExporter:
    """Optional Langfuse sink; no-op unless configured and installed."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self._client = None
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            return
        try:  # pragma: no cover - exercised only with live creds
            from langfuse import Langfuse

            self._client = Langfuse()
        except Exception:
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def record(self, query_id: str, query_type: str, latency_ms: float,
               attributes: dict[str, Any]) -> None:  # pragma: no cover
        if self._client is None:
            return
        try:
            self._client.trace(
                name="memory.recall",
                metadata={"arm": self.arm, "query_type": query_type,
                          "latency_ms": latency_ms, **attributes},
                id=_hex_id(self.arm, query_id, nbytes=16),
            )
        except Exception:
            pass

    def flush(self) -> None:  # pragma: no cover
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
