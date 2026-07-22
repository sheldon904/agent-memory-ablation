"""Vector index: sqlite-vec store + exact-cosine search.

The corpus vectors are persisted into a ``sqlite-vec`` (vec0) virtual table so
(a) the RAG arm's store is the spec-named embedded index and (b) storage bytes
are a real on-disk measurement. Retrieval itself is exact brute-force cosine
over the in-memory normalised matrix: at ~1,600 vectors this is a few
milliseconds and is *identical in result* to vec0's own top-k (vec0 is exact at
this scale), while giving clean cosine scores for the 0.40 threshold. This
mirrors the finding in the system under ablation that approximate indexing buys
nothing at a solo operator's store size.
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np

try:
    import sqlite_vec  # type: ignore

    _HAVE_VEC = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_VEC = False


class VectorIndex:
    def __init__(self, dim: int, db_path: str) -> None:
        self.dim = dim
        self.db_path = db_path
        self.ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._id_pos: dict[str, int] = {}
        self.backend = "unset"

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        assert vectors.shape[0] == len(ids)
        assert vectors.shape[1] == self.dim
        self.ids = list(ids)
        self._matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        self._id_pos = {i: p for p, i in enumerate(ids)}

        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        if _HAVE_VEC:
            con = sqlite3.connect(self.db_path)
            try:
                con.enable_load_extension(True)
                sqlite_vec.load(con)
                con.enable_load_extension(False)
                con.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING vec0("
                    f"embedding float[{self.dim}])"
                )
                con.execute("CREATE TABLE ids(rowid INTEGER PRIMARY KEY, fact_id TEXT)")
                with con:
                    for rowid, (fid, vec) in enumerate(zip(ids, vectors), start=1):
                        con.execute(
                            "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                            (rowid, vec.astype(np.float32).tobytes()),
                        )
                        con.execute(
                            "INSERT INTO ids(rowid, fact_id) VALUES (?, ?)",
                            (rowid, fid),
                        )
                self.backend = "sqlite-vec"
            finally:
                con.close()
        else:  # hermetic fallback: store raw vectors as BLOBs
            con = sqlite3.connect(self.db_path)
            try:
                con.execute(
                    "CREATE TABLE vec_items(rowid INTEGER PRIMARY KEY, "
                    "fact_id TEXT, embedding BLOB)"
                )
                with con:
                    for rowid, (fid, vec) in enumerate(zip(ids, vectors), start=1):
                        con.execute(
                            "INSERT INTO vec_items VALUES (?, ?, ?)",
                            (rowid, fid, vec.astype(np.float32).tobytes()),
                        )
                self.backend = "sqlite-blob"
            finally:
                con.close()

    def query(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the top-k ``(fact_id, cosine)`` for a unit query vector."""
        if self._matrix.shape[0] == 0:
            return []
        sims = self._matrix @ vec.astype(np.float32)
        k = min(k, sims.shape[0])
        # argpartition for the top-k, then sort those descending
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], float(sims[i])) for i in idx]

    def storage_bytes(self) -> int:
        return os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
