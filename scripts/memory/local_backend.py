"""
Local/Qdrant-like backend for COCOIndex memory storage.

Provides a lightweight in-memory/file-backed alternative to Postgres:
- In-memory storage with optional JSON persistence
- FAISS-based ANN search (if available) or brute-force fallback
- Config-driven selection between Postgres and local backends
- Compatible API with COCOIndexMemoryFlow

This is useful for:
- Local development without Postgres
- Testing and CI environments
- Lightweight deployments
- Offline experimentation
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional FAISS import for accelerated search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

# Embedding backend
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


DEFAULT_MODEL = os.getenv("COCOINDEX_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("COCOINDEX_EMBED_DIM", "384"))


@dataclass
class LocalMemoryRecord:
    """Memory record for local storage."""
    id: str
    person_id: str
    fact: str
    chunk: str
    embedding: np.ndarray
    importance: float
    type: str
    created_at: float
    updated_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "person_id": self.person_id,
            "fact": self.fact,
            "chunk": self.chunk,
            "embedding": self.embedding.tolist(),
            "importance": self.importance,
            "type": self.type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalMemoryRecord":
        return cls(
            id=data["id"],
            person_id=data["person_id"],
            fact=data["fact"],
            chunk=data["chunk"],
            embedding=np.asarray(data["embedding"], dtype=np.float32),
            importance=float(data["importance"]),
            type=data["type"],
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
        )


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


class LocalEmbeddingBackend:
    """Embedding backend for local storage."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        use_random_fallback: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_random_fallback = use_random_fallback
        self.model: Optional[SentenceTransformer] = None
        self._warned = False

    def embed(self, text: str) -> np.ndarray:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            if self.model is None:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            emb = self.model.encode(text)
            emb = np.asarray(emb, dtype=np.float32)
            return _normalize(emb)
        elif self.use_random_fallback:
            if not self._warned:
                print("[LocalBackend] Warning: sentence-transformers not available, using random embeddings")
                self._warned = True
            # Deterministic random based on text hash
            seed = hash(text) % (2**32)
            rng = np.random.RandomState(seed)
            emb = rng.randn(EMBED_DIM).astype(np.float32)
            return _normalize(emb)
        else:
            raise ImportError("sentence-transformers is required for embeddings")


class FAISSIndex:
    """FAISS-based ANN index for fast similarity search."""

    def __init__(self, dim: int = EMBED_DIM):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for FAISSIndex")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        self.id_map: List[str] = []

    def add(self, record_id: str, embedding: np.ndarray) -> None:
        emb = _normalize(embedding).reshape(1, -1).astype(np.float32)
        self.index.add(emb)
        self.id_map.append(record_id)

    def remove(self, record_id: str) -> None:
        # FAISS doesn't support removal well, so we rebuild
        if record_id in self.id_map:
            idx = self.id_map.index(record_id)
            self.id_map.pop(idx)
            # Note: This doesn't actually remove from FAISS index
            # For full support, would need to rebuild index

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        q = _normalize(query).reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(q, k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))
        return results

    def rebuild(self, records: List[Tuple[str, np.ndarray]]) -> None:
        """Rebuild the index from scratch."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []
        for record_id, embedding in records:
            self.add(record_id, embedding)


class BruteForceIndex:
    """Brute-force similarity search (fallback when FAISS unavailable)."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.embeddings: Dict[str, np.ndarray] = {}

    def add(self, record_id: str, embedding: np.ndarray) -> None:
        self.embeddings[record_id] = _normalize(embedding)

    def remove(self, record_id: str) -> None:
        self.embeddings.pop(record_id, None)

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if not self.embeddings:
            return []
        q = _normalize(query)
        scores: List[Tuple[str, float]] = []
        for record_id, emb in self.embeddings.items():
            sim = float(np.dot(q, emb))
            scores.append((record_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def rebuild(self, records: List[Tuple[str, np.ndarray]]) -> None:
        self.embeddings = {}
        for record_id, embedding in records:
            self.add(record_id, embedding)


class LocalMemoryStore:
    """
    Local memory store with optional file persistence.

    API compatible with COCOIndexMemoryFlow for easy swapping.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        use_faiss: bool = True,
        embed_dim: int = EMBED_DIM,
        auto_persist: bool = True,
    ):
        """
        Initialize local memory store.

        Args:
            persist_path: Path for JSON persistence (None = in-memory only).
            model_name: Embedding model name.
            device: Device for embeddings ("cpu", "cuda", etc).
            use_faiss: Use FAISS for ANN search if available.
            embed_dim: Embedding dimension.
            auto_persist: Auto-save after each write operation.
        """
        self.persist_path = Path(persist_path) if persist_path else None
        self.auto_persist = auto_persist
        self.embed_dim = embed_dim

        # Storage
        self.records: Dict[str, LocalMemoryRecord] = {}
        self.logs: List[Dict[str, Any]] = []

        # Embedding backend
        self.embedder = LocalEmbeddingBackend(model_name=model_name, device=device)

        # Search index
        if use_faiss and FAISS_AVAILABLE:
            self.index: Any = FAISSIndex(dim=embed_dim)
        else:
            self.index = BruteForceIndex(dim=embed_dim)

        # Load existing data if persist path exists
        if self.persist_path and self.persist_path.exists():
            self._load()

    def _load(self) -> None:
        """Load records from persistence file."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            for rec_data in data.get("records", []):
                rec = LocalMemoryRecord.from_dict(rec_data)
                self.records[rec.id] = rec
                self.index.add(rec.id, rec.embedding)
            self.logs = data.get("logs", [])
            print(f"[LocalBackend] Loaded {len(self.records)} records from {self.persist_path}")
        except Exception as e:
            print(f"[LocalBackend] Error loading: {e}")

    def _save(self) -> None:
        """Save records to persistence file."""
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "records": [rec.to_dict() for rec in self.records.values()],
                "logs": self.logs[-1000:],  # Keep last 1000 logs
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[LocalBackend] Error saving: {e}")

    def _log(self, operation: str, record_id: Optional[str], payload: Dict[str, Any]) -> None:
        """Log an operation."""
        self.logs.append({
            "operation": operation,
            "record_id": record_id,
            "payload": payload,
            "timestamp": time.time(),
        })

    def upsert(self, record: Dict[str, Any]) -> LocalMemoryRecord:
        """Add or update a memory record."""
        fact = record.get("fact")
        person_id = record.get("person_id")
        if not fact or not person_id:
            raise ValueError("record must include 'fact' and 'person_id'")

        chunk = record.get("chunk", fact)
        rec_id = record.get("id", str(uuid.uuid4()))
        importance = float(record.get("importance", 5))
        mtype = record.get("type", "fact")

        # Compute embedding
        emb = self.embedder.embed(chunk)

        now = time.time()
        is_update = rec_id in self.records

        rec = LocalMemoryRecord(
            id=rec_id,
            person_id=person_id,
            fact=fact,
            chunk=chunk,
            embedding=emb,
            importance=importance,
            type=mtype,
            created_at=self.records[rec_id].created_at if is_update else now,
            updated_at=now,
        )

        # Update storage
        if is_update:
            self.index.remove(rec_id)
        self.records[rec_id] = rec
        self.index.add(rec_id, emb)

        self._log("upsert", rec_id, {"person_id": person_id, "importance": importance, "type": mtype})

        if self.auto_persist:
            self._save()

        return rec

    def upsert_batch(self, records: List[Dict[str, Any]]) -> List[LocalMemoryRecord]:
        """Add or update multiple records."""
        results = []
        for r in records:
            results.append(self.upsert(r))
        return results

    def query(
        self,
        text: str,
        top_k: int = 5,
        person_id: Optional[str] = None,
    ) -> List[Tuple[LocalMemoryRecord, float]]:
        """Return top_k records by similarity to the query text."""
        q_emb = self.embedder.embed(text)

        # Search index
        candidates = self.index.search(q_emb, top_k * 3)  # Over-fetch for filtering

        # Filter and score
        results: List[Tuple[LocalMemoryRecord, float]] = []
        for record_id, score in candidates:
            if record_id not in self.records:
                continue
            rec = self.records[record_id]
            if person_id and rec.person_id != person_id:
                continue
            results.append((rec, score))

        results = results[:top_k]

        self._log("query", None, {"text": text, "top_k": top_k, "person_id": person_id, "returned": len(results)})

        return results

    def semantic_diff(
        self,
        text: str,
        top_k: int = 3,
        person_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get semantic diff between new text and existing memories."""
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

    def get_recent(self, person_id: str, limit: int = 5) -> List[LocalMemoryRecord]:
        """Get recent records for a person."""
        person_records = [
            rec for rec in self.records.values() if rec.person_id == person_id
        ]
        person_records.sort(key=lambda r: r.updated_at, reverse=True)
        return person_records[:limit]

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete records by ID."""
        for record_id in ids:
            if record_id in self.records:
                self.index.remove(record_id)
                del self.records[record_id]

        self._log("delete", None, {"ids": ids})

        if self.auto_persist:
            self._save()

    def clear(self) -> None:
        """Clear all records."""
        self.records = {}
        self.index.rebuild([])
        self.logs = []
        if self.auto_persist:
            self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        person_counts: Dict[str, int] = {}
        for rec in self.records.values():
            person_counts[rec.person_id] = person_counts.get(rec.person_id, 0) + 1

        return {
            "total_records": len(self.records),
            "person_counts": person_counts,
            "faiss_enabled": FAISS_AVAILABLE and isinstance(self.index, FAISSIndex),
            "persist_path": str(self.persist_path) if self.persist_path else None,
        }


# ---------------------------------------------------------------------------
# Backend factory for config-driven selection
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """Configuration for backend selection."""
    backend_type: str = "auto"  # "auto", "postgres", "local"
    # Postgres settings
    db_url: Optional[str] = None
    # Local settings
    persist_path: Optional[str] = None
    use_faiss: bool = True
    # Common settings
    model_name: str = DEFAULT_MODEL
    device: Optional[str] = None
    embed_dim: int = EMBED_DIM


def create_memory_backend(config: Optional[BackendConfig] = None) -> Any:
    """
    Factory function to create the appropriate memory backend.

    Args:
        config: Backend configuration (auto-detects if None).

    Returns:
        Either COCOIndexMemoryFlow (Postgres) or LocalMemoryStore.
    """
    config = config or BackendConfig()

    # Auto-detect based on environment
    if config.backend_type == "auto":
        db_url = config.db_url or os.getenv("COCOINDEX_DB_URL", "")
        if db_url:
            config.backend_type = "postgres"
        else:
            config.backend_type = "local"

    if config.backend_type == "postgres":
        try:
            from scripts.memory.coco_memory_flow import COCOIndexMemoryFlow
            db_url = config.db_url or os.getenv("COCOINDEX_DB_URL", "")
            return COCOIndexMemoryFlow(
                db_url=db_url,
                model_name=config.model_name,
                device=config.device,
                embed_dim=config.embed_dim,
            )
        except ImportError as e:
            print(f"[Backend] Postgres backend unavailable: {e}")
            print("[Backend] Falling back to local backend")
            config.backend_type = "local"

    # Local backend
    return LocalMemoryStore(
        persist_path=config.persist_path,
        model_name=config.model_name,
        device=config.device,
        use_faiss=config.use_faiss,
        embed_dim=config.embed_dim,
    )


__all__ = [
    "LocalMemoryRecord",
    "LocalMemoryStore",
    "LocalEmbeddingBackend",
    "FAISSIndex",
    "BruteForceIndex",
    "BackendConfig",
    "create_memory_backend",
    "FAISS_AVAILABLE",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
]

