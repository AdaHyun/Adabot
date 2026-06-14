"""
Persistent memory storage for Adabot.

The public API deliberately hides SQLite and Chroma behind a small interface so
the app can migrate to PostgreSQL + pgvector later without touching main.py.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
MEMORY_DIR = PROJECT_ROOT / "memory"
SQLITE_PATH = MEMORY_DIR / "memory.db"
CHROMA_DIR = MEMORY_DIR / "chroma"
COLLECTION_NAME = "adabot_memory"
EMBEDDING_DIM = 384


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dirs() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            source TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memories_user_type
        ON memories(user_id, memory_type)
        """
    )
    return conn


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_./:\\-]+|[\u4e00-\u9fff]", (text or "").lower())


def _embed(text: str) -> List[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _collection():
    try:
        import chromadb
    except ImportError:
        return None

    _ensure_dirs()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _sqlite_search(user_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    query_tokens = set(_tokens(query))
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, content, memory_type, importance, source, created_at, updated_at
            FROM memories
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()

    scored: List[Dict[str, Any]] = []
    for row in rows:
        content_tokens = set(_tokens(row["content"]))
        overlap = len(query_tokens & content_tokens)
        score = (overlap / max(len(query_tokens), 1)) + float(row["importance"]) * 0.05
        if overlap or not query_tokens:
            item = dict(row)
            item["score"] = round(score, 4)
            scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def add_memory(
    user_id: str,
    content: str,
    memory_type: str,
    importance: float = 0.5,
    source: str = "",
) -> Dict[str, Any]:
    """Add or refresh one memory item."""
    normalized_content = " ".join((content or "").split())
    if not normalized_content:
        raise ValueError("memory content cannot be empty")

    now = _now()
    with _connect() as conn:
        existing = conn.execute(
            """
            SELECT id, user_id, content, memory_type, importance, source, created_at, updated_at
            FROM memories
            WHERE user_id = ? AND memory_type = ? AND content = ?
            """,
            (user_id, memory_type, normalized_content),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE memories
                SET importance = MAX(importance, ?), updated_at = ?
                WHERE id = ?
                """,
                (importance, now, existing["id"]),
            )
            memory_id = existing["id"]
            created_at = existing["created_at"]
        else:
            memory_id = uuid.uuid4().hex
            created_at = now
            conn.execute(
                """
                INSERT INTO memories
                (id, user_id, content, memory_type, importance, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (memory_id, user_id, normalized_content, memory_type, importance, source, created_at, now),
            )

    collection = _collection()
    if collection is not None:
        collection.upsert(
            ids=[memory_id],
            documents=[normalized_content],
            metadatas=[{"user_id": user_id, "memory_type": memory_type, "source": source}],
            embeddings=[_embed(normalized_content)],
        )

    return {
        "id": memory_id,
        "user_id": user_id,
        "content": normalized_content,
        "memory_type": memory_type,
        "importance": importance,
        "source": source,
        "created_at": created_at,
        "updated_at": now,
    }


def search_memory(user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search memories for a user. Chroma is preferred; SQLite lexical search is the fallback."""
    collection = _collection()
    if collection is None:
        return _sqlite_search(user_id, query, top_k)

    result = collection.query(
        query_embeddings=[_embed(query)],
        n_results=top_k,
        where={"user_id": user_id},
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    if not ids:
        return []

    with _connect() as conn:
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""
            SELECT id, user_id, content, memory_type, importance, source, created_at, updated_at
            FROM memories
            WHERE id IN ({placeholders})
            """,
            ids,
        ).fetchall()

    by_id = {row["id"]: dict(row) for row in rows}
    items: List[Dict[str, Any]] = []
    for index, memory_id in enumerate(ids):
        item = by_id.get(memory_id, {})
        if not item:
            item = {
                "id": memory_id,
                "user_id": user_id,
                "content": documents[index],
                "memory_type": metadatas[index].get("memory_type", "general"),
                "source": metadatas[index].get("source", ""),
                "importance": 0.5,
            }
        distance = float(distances[index]) if index < len(distances) else 1.0
        item["score"] = round(max(0.0, 1.0 - distance), 4)
        items.append(item)

    return items


def list_memory(user_id: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, content, memory_type, importance, source, created_at, updated_at
            FROM memories
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_memory(memory_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    collection = _collection()
    if collection is not None:
        collection.delete(ids=[memory_id])

    return cursor.rowcount > 0
