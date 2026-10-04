"""Lazy BM25 search over the indexed chunks, using Python's bundled SQLite FTS5."""

import re
import sqlite3
import threading
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document

_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "use",
        "using",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "you",
    ]
)


def chunk_key(document: Document) -> str:
    """Identify a chunk without depending on its rank in either search."""
    return str(document.metadata.get("chunk_id") or document.id or document.page_content)


class KeywordSearch:
    """Build one thread-safe, in-memory lexical index from the same Chroma snapshot.

    No embeddings or additional model are loaded. Rebuild the graph after ingestion to
    refresh the snapshot, just as for its cached vector-store provider.
    """

    def __init__(self, store: Callable[[], Any]) -> None:
        """Keep a lazy store provider; no index is opened during construction."""
        self._store = store
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        self._documents: list[Document] = []

    def __call__(self, query: str, k: int) -> list[Document]:
        """Return BM25-ranked matches; quote tokens so user input cannot become FTS syntax."""
        tokens = list(dict.fromkeys(re.findall(r"[:$#]?\w[\w-]*", query.casefold())))
        tokens = [token for token in tokens if token not in _STOP_WORDS and len(token) > 1]
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens[:64])
        with self._lock:
            if self._connection is None:
                self._build()
            assert self._connection is not None
            rows = self._connection.execute(
                "SELECT rowid FROM chunks WHERE chunks MATCH ? "
                "ORDER BY bm25(chunks, 3.0, 1.0), rowid LIMIT ?",
                (expression, k),
            ).fetchall()
            return [self._documents[row[0] - 1] for row in rows]

    def _build(self) -> None:
        """Read Chroma in pages; publish the completed index only after a successful build."""
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        documents: list[Document] = []
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE chunks USING fts5(title, content, "
                "tokenize = \"unicode61 tokenchars '_:$#-'\")"
            )
            offset = 0
            while True:
                page = self._store().get(
                    limit=1000, offset=offset, include=["documents", "metadatas"]
                )
                if not page["ids"]:
                    break
                for identifier, content, metadata in zip(
                    page["ids"], page["documents"], page["metadatas"], strict=True
                ):
                    metadata = metadata or {}
                    documents.append(
                        Document(id=identifier, page_content=content, metadata=metadata)
                    )
                    connection.execute(
                        "INSERT INTO chunks(title, content) VALUES (?, ?)",
                        (metadata.get("title", "") + " " + metadata.get("section", ""), content),
                    )
                offset += len(page["ids"])
            connection.commit()
        except BaseException:
            connection.close()
            raise
        self._documents = documents
        self._connection = connection


def fuse_rankings(
    semantic: Sequence[tuple[Document, float]], lexical: Sequence[Document], k: int
) -> list[tuple[Document, float]]:
    """Reciprocal-rank fusion; retain cosine scores separately from ranking scores.

    A cosine score of -1 denotes a lexical-only hit, not an embedding similarity.
    """
    documents = {chunk_key(doc): (doc, -1.0) for doc in lexical}
    documents.update({chunk_key(doc): (doc, score) for doc, score in semantic})
    ranks: dict[str, float] = {}
    for ranking in ([doc for doc, _ in semantic], lexical):
        seen: set[str] = set()
        for rank, doc in enumerate(ranking, start=1):
            key = chunk_key(doc)
            if key not in seen:
                ranks[key] = ranks.get(key, 0.0) + 1.0 / (60 + rank)
                seen.add(key)
    return [documents[key] for key in sorted(ranks, key=lambda key: (-ranks[key], key))[:k]]
