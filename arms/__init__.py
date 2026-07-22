"""Three memory arms behind one ``MemoryProvider`` interface."""

from .base import MemoryProvider, Recall, RecallResponse, load_corpus
from .rag import StatelessRAG
from .graph import GraphOnly
from .hybrid import Hybrid


def make_arm(name: str, index_dir: str, **kw) -> MemoryProvider:
    """Factory: 'rag' | 'graph' | 'hybrid'."""
    name = name.lower()
    if name == "rag":
        return StatelessRAG(index_dir, **kw)
    if name == "graph":
        return GraphOnly(index_dir, **kw)
    if name == "hybrid":
        return Hybrid(index_dir, **kw)
    raise ValueError(f"unknown arm: {name!r}")


ARMS = ("rag", "graph", "hybrid")

__all__ = [
    "MemoryProvider", "Recall", "RecallResponse", "load_corpus",
    "StatelessRAG", "GraphOnly", "Hybrid", "make_arm", "ARMS",
]
