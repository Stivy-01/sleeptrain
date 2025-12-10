"""
COCOIndex-style memory flow with Postgres persistence and real embeddings.

Requirements:
- Environment: COCOINDEX_DB_URL (Postgres URL), optional COCOINDEX_EMBED_MODEL.
- sentence-transformers installed (no fallback embeddings).
- Optional pgvector extension: used automatically if available for in-DB ANN.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import psycopg2
    from psycopg2.extras import Json
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("psycopg2 is required for COCOIndexMemoryFlow") from exc

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("sentence-transformers is required for COCOIndexMemoryFlow") from exc


DEFAULT_MODEL = os.getenv("COCOINDEX_EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_DB_URL = os.getenv("COCOINDEX_DB_URL", "")
EMBED_DIM = int(os.getenv("COCOINDEX_EMBED_DIM", "384"))


def _vector_literal(emb: np.ndarray) -> str:
    """Convert an embedding to pgvector literal text."""
    return "[" + ",".join(f"{x:.6f}" for x in emb.tolist()) + "]"


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


@dataclass
class MemoryRecord:
    id: str
    person_id: str
    fact: str
    chunk: str
    embedding: np.ndarray
    importance: float
    type: str
    created_at: float
    updated_at: float


class _DB:
    """Thin Postgres helper with optional pgvector usage."""

    def __init__(self, db_url: str, embed_dim: int = EMBED_DIM):
        if not db_url:
            raise ValueError("COCOINDEX_DB_URL is required")
        self.db_url = db_url
        self.embed_dim = embed_dim
        self.pgvector_enabled = False
        self._ensure_schema()

    def _connect(self):
        return psycopg2.connect(self.db_url)

    def _execute(self, sql: str, params: Sequence[Any] = (), fetch: Optional[str] = None):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                return None

    def _detect_pgvector(self) -> bool:
        res = self._execute(
            "SELECT 1 FROM pg_extension WHERE extname='vector';",
            fetch="one",
        )
        return bool(res)

    def _ensure_schema(self):
        self.pgvector_enabled = self._detect_pgvector()

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                id UUID PRIMARY KEY,
                person_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                chunk TEXT NOT NULL,
                embedding DOUBLE PRECISION[] NOT NULL,
                importance DOUBLE PRECISION NOT NULL,
                type TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_records_person ON memory_records(person_id);"
        )
        if self.pgvector_enabled:
            self._execute(
                f"ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS embedding_vec vector({self.embed_dim});"
            )
            self._execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_records_embedding_vec ON memory_records USING ivfflat (embedding_vec vector_cosine_ops);"
            )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS memory_logs (
                id BIGSERIAL PRIMARY KEY,
                operation TEXT NOT NULL,
                record_id UUID,
                payload JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )

    def upsert_record(
        self,
        rec_id: str,
        person_id: str,
        fact: str,
        chunk: str,
        embedding: np.ndarray,
        importance: float,
        mtype: str,
        embed_vec_str: Optional[str],
    ):
        if self.pgvector_enabled:
            sql = """
            INSERT INTO memory_records (id, person_id, fact, chunk, embedding, embedding_vec, importance, type, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, now(), now())
            ON CONFLICT (id) DO UPDATE
            SET fact=EXCLUDED.fact,
                chunk=EXCLUDED.chunk,
                embedding=EXCLUDED.embedding,
                embedding_vec=EXCLUDED.embedding_vec,
                importance=EXCLUDED.importance,
                type=EXCLUDED.type,
                updated_at=now();
            """
            params = (
                rec_id,
                person_id,
                fact,
                chunk,
                embedding.tolist(),
                embed_vec_str,
                importance,
                mtype,
            )
        else:
            sql = """
            INSERT INTO memory_records (id, person_id, fact, chunk, embedding, importance, type, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())
            ON CONFLICT (id) DO UPDATE
            SET fact=EXCLUDED.fact,
                chunk=EXCLUDED.chunk,
                embedding=EXCLUDED.embedding,
                importance=EXCLUDED.importance,
                type=EXCLUDED.type,
                updated_at=now();
            """
            params = (
                rec_id,
                person_id,
                fact,
                chunk,
                embedding.tolist(),
                importance,
                mtype,
            )
        self._execute(sql, params)

    def fetch_candidates(
        self,
        person_id: Optional[str],
        vector_str: Optional[str],
        limit: int,
        use_pgvector: bool,
    ) -> List[Tuple]:
        where = []
        params: List[Any] = []
        if person_id:
            where.append("person_id = %s")
            params.append(person_id)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        if use_pgvector and self.pgvector_enabled and vector_str:
            sql = f"""
                SELECT id, person_id, fact, chunk, embedding, importance, type, created_at, updated_at
                FROM memory_records
                {where_sql}
                ORDER BY embedding_vec <-> %s::vector
                LIMIT %s;
            """
            params = params + [vector_str, limit]
        else:
            sql = f"""
                SELECT id, person_id, fact, chunk, embedding, importance, type, created_at, updated_at
                FROM memory_records
                {where_sql}
                ORDER BY updated_at DESC
                LIMIT %s;
            """
            params = params + [limit]

        return self._execute(sql, params, fetch="all") or []

    def log(self, operation: str, record_id: Optional[str], payload: Dict[str, Any]):
        self._execute(
            """
            INSERT INTO memory_logs (operation, record_id, payload, created_at)
            VALUES (%s, %s, %s, now());
            """,
            (operation, record_id, Json(payload)),
        )

    def delete_records(self, ids: List[str]) -> None:
        if not ids:
            return
        placeholders = ",".join(["%s"] * len(ids))
        self._execute(f"DELETE FROM memory_records WHERE id IN ({placeholders});", ids)


class EmbeddingBackend:
    """Real embedding backend (sentence-transformers only)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model: Optional[SentenceTransformer] = None

    def embed(self, text: str) -> np.ndarray:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        emb = self.model.encode(text)
        emb = np.asarray(emb, dtype=np.float32)
        return _normalize(emb)


class COCOIndexMemoryFlow:
    """
    COCOIndex-style memory store backed by Postgres and real embeddings.

    API:
        - upsert(record_dict) -> MemoryRecord
        - upsert_batch(records) -> List[MemoryRecord]
        - query(text, top_k=5, person_id=None) -> List[Tuple[MemoryRecord, float]]
        - semantic_diff(text, person_id=None) -> Dict
        - get_recent(person_id, limit=5) -> List[MemoryRecord]
        - delete_by_ids(ids) -> None (test helper)
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        enable_pgvector: bool = True,
        embed_dim: int = EMBED_DIM,
    ):
        self.db = _DB(db_url or DEFAULT_DB_URL, embed_dim=embed_dim)
        # honor caller preference but also respect db capability
        self.use_pgvector = enable_pgvector and self.db.pgvector_enabled
        self.embedder = EmbeddingBackend(model_name=model_name, device=device)

    # --------------------
    # Core operations
    # --------------------
    def upsert(self, record: Dict[str, Any]) -> MemoryRecord:
        """
        Add or update a memory record.

        Required keys:
            person_id (str): identifier for the person/entity
            fact (str): full fact text
        Optional keys:
            chunk (str): chunked/processed text (default = fact)
            importance (float): 0-10 scale (default 5)
            type (str): metadata label (default "fact")
            id (str): if provided, replaces existing with same id
        """
        fact = record.get("fact")
        person_id = record.get("person_id")
        if not fact or not person_id:
            raise ValueError("record must include 'fact' and 'person_id'")

        chunk = record.get("chunk", fact)
        rec_id = record.get("id", str(uuid.uuid4()))
        importance = float(record.get("importance", 5))
        mtype = record.get("type", "fact")

        emb = self.embedder.embed(chunk)
        embed_vec_str = _vector_literal(emb) if self.use_pgvector else None

        self.db.upsert_record(
            rec_id=rec_id,
            person_id=person_id,
            fact=fact,
            chunk=chunk,
            embedding=emb,
            importance=importance,
            mtype=mtype,
            embed_vec_str=embed_vec_str,
        )
        self.db.log(
            "upsert",
            rec_id,
            {"person_id": person_id, "importance": importance, "type": mtype},
        )
        return MemoryRecord(
            id=rec_id,
            person_id=person_id,
            fact=fact,
            chunk=chunk,
            embedding=emb,
            importance=importance,
            type=mtype,
            created_at=time.time(),
            updated_at=time.time(),
        )

    def upsert_batch(self, records: List[Dict[str, Any]]) -> List[MemoryRecord]:
        return [self.upsert(r) for r in records]

    def query(
        self,
        text: str,
        top_k: int = 5,
        person_id: Optional[str] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Return top_k records by cosine similarity to the query text."""
        q_emb = self.embedder.embed(text)
        candidates = self.db.fetch_candidates(
            person_id=person_id,
            vector_str=_vector_literal(q_emb) if self.use_pgvector else None,
            limit=max(top_k * 3, top_k),
            use_pgvector=self.use_pgvector,
        )
        results: List[Tuple[MemoryRecord, float]] = []
        for row in candidates:
            (
                rec_id,
                rec_person,
                fact,
                chunk,
                embedding_arr,
                importance,
                mtype,
                created_at,
                updated_at,
            ) = row
            emb = np.asarray(embedding_arr, dtype=np.float32)
            sim = float(np.dot(q_emb, emb))
            results.append(
                (
                    MemoryRecord(
                        id=str(rec_id),
                        person_id=rec_person,
                        fact=fact,
                        chunk=chunk,
                        embedding=emb,
                        importance=float(importance),
                        type=mtype,
                        created_at=float(created_at.timestamp()),
                        updated_at=float(updated_at.timestamp()),
                    ),
                    sim,
                )
            )
        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]

        self.db.log(
            "query",
            None,
            {
                "person_id": person_id,
                "text": text,
                "top_k": top_k,
                "returned": len(top),
            },
        )
        return top

    # --------------------
    # Diff / diagnostics
    # --------------------
    def semantic_diff(
        self, text: str, top_k: int = 3, person_id: Optional[str] = None
    ) -> Dict[str, Any]:
        matches = self.query(text, top_k=top_k, person_id=person_id)
        response = {
            "new_text": text,
            "matches": [
                {
                    "id": rec.id,
                    "fact": rec.fact,
                    "person_id": rec.person_id,
                    "type": rec.type,
                    "importance": rec.importance,
                    "score": score,
                }
                for rec, score in matches
            ],
        }
        if matches:
            best, score = matches[0]
            response["best_match_id"] = best.id
            response["best_match_score"] = score
        else:
            response["best_match_id"] = None
            response["best_match_score"] = None
        return response

    def get_recent(self, person_id: str, limit: int = 5) -> List[MemoryRecord]:
        rows = self.db.fetch_candidates(
            person_id=person_id,
            vector_str=None,
            limit=limit,
            use_pgvector=False,
        )
        records: List[MemoryRecord] = []
        for row in rows:
            (
                rec_id,
                rec_person,
                fact,
                chunk,
                embedding_arr,
                importance,
                mtype,
                created_at,
                updated_at,
            ) = row
            records.append(
                MemoryRecord(
                    id=str(rec_id),
                    person_id=rec_person,
                    fact=fact,
                    chunk=chunk,
                    embedding=np.asarray(embedding_arr, dtype=np.float32),
                    importance=float(importance),
                    type=mtype,
                    created_at=float(created_at.timestamp()),
                    updated_at=float(updated_at.timestamp()),
                )
            )
        return records

    # --------------------
    # Test / maintenance helpers
    # --------------------
    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete records by id (used in tests/cleanup)."""
        self.db.delete_records(ids)


__all__ = ["COCOIndexMemoryFlow", "MemoryRecord", "EMBED_DIM", "DEFAULT_MODEL", "DEFAULT_DB_URL"]
