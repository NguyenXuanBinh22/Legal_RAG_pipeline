"""chunks.db: chunk + metadata trên đĩa, đúng schema mà SearchService (services/Search.py) đọc.

Đây là nguồn dữ liệu DUY NHẤT giữa Phase 3 và Phase 4: faiss_id = số thứ tự chunk (0, 1, 2…)
và cũng là id của vector trong FAISS index. Thay cho chunks.json + faiss_id_map.json +
chunk_map.json + doc_index_map.json + article_index_map.json của chunk_embedding.py cũ
(SearchService chỉ đọc chunks.db, và tra cứu theo doc_id/Điều đã có index SQLite).
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE chunks (
    faiss_id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    doc_id   TEXT,
    doc_num  TEXT,
    title    TEXT,
    article  TEXT,
    clause   TEXT,
    content  TEXT
);
CREATE TABLE docs (
    doc_id       TEXT PRIMARY KEY,
    doc_num      TEXT,
    title        TEXT,
    doc_num_norm TEXT,
    title_norm   TEXT
);
"""
INDEXES = (
    "CREATE INDEX idx_chunks_doc_id ON chunks(doc_id)",
    "CREATE INDEX idx_docs_doc_num_norm ON docs(doc_num_norm)",
)
INSERT_BATCH_SIZE = 10_000


def normalize_doc_ref(text: str | None) -> str:
    """Phải khớp với SearchService._normalize_doc_ref"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


class ChunkDbWriter:
    """Ghi chunks.db mới vào file .tmp, os.replace khi thành công. Lỗi giữa chừng → file cũ còn nguyên."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.count = 0
        self._rows: list[tuple] = []

    def __enter__(self) -> "ChunkDbWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp.unlink(missing_ok=True)
        self._db = sqlite3.connect(self.tmp)
        self._db.execute("PRAGMA journal_mode = OFF")
        self._db.execute("PRAGMA synchronous = OFF")
        self._db.executescript(SCHEMA)
        return self

    def add_doc(self, doc_id: str, doc_num: str | None, title: str | None) -> None:
        self._db.execute(
            "INSERT INTO docs VALUES (?, ?, ?, ?, ?)",
            (doc_id, doc_num, title, normalize_doc_ref(doc_num), normalize_doc_ref(title)),
        )

    def add_chunk(self, chunk_id: str, doc_id: str, doc_num: str | None, title: str | None,
                  article: str | None, clause: str | None, content: str) -> int:
        faiss_id = self.count
        self._rows.append((faiss_id, chunk_id, doc_id, doc_num, title, article, clause, content))
        self.count += 1
        if len(self._rows) >= INSERT_BATCH_SIZE:
            self._flush()
        return faiss_id

    def _flush(self) -> None:
        self._db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", self._rows)
        self._rows.clear()

    def __exit__(self, exc_type, *_) -> None:
        try:
            if exc_type is None:
                self._flush()
                for sql in INDEXES:
                    self._db.execute(sql)
                self._db.commit()
        finally:
            self._db.close()
        if exc_type is None:
            os.replace(self.tmp, self.path)
        else:
            self.tmp.unlink(missing_ok=True)


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def count_chunks(path: Path) -> int:
    # `with sqlite3.connect()` chỉ commit, không đóng kết nối
    with closing(_connect_ro(path)) as db:
        return db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def iter_chunk_texts(path: Path) -> Iterator[tuple[int, str]]:
    """(faiss_id, content) theo thứ tự faiss_id; content chính là embed_text."""
    with closing(_connect_ro(path)) as db:
        yield from db.execute("SELECT faiss_id, content FROM chunks ORDER BY faiss_id")
