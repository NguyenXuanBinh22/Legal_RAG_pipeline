"""
Phase 4: chunks.db → faiss.index qua API embedding kiểu OpenAI (LM Studio…). Resume được.

Trạng thái resume nằm trong chính index (id_map = các faiss_id đã embed) và embed_skipped.json
(chunk vượt context của model), không còn indexed_ids.json ghi tách rời index như code cũ.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests

from .chunk_db import count_chunks, iter_chunk_texts
from .config import USER_AGENT
from .storage import atomic_write_json
from .vbpl_client import make_session
from .vector_index import IndexMismatch, indexed_ids, l2_normalize, new_flat_index, read_index, write_index_atomic

log = logging.getLogger(__name__)

# Lỗi vượt context có thể đến dưới nhiều mã HTTP tuỳ server, nên nhận diện theo nội dung
CONTEXT_ERROR_MARKERS = ("context length", "input_tokens", "too long")
LOG_EVERY = 1000


class EmbeddingError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class TextTooLong(EmbeddingError):
    """Văn bản vượt context của model embedding: bỏ qua chunk, không dừng cả lần chạy."""


def _error_message(resp) -> str:
    try:
        body = resp.json()
    except ValueError:
        return str(getattr(resp, "text", ""))
    err = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(err, dict):
        err = err.get("message", err)
    return str(err)


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: tuple[float, float] = (5.0, 300.0),
        retries: int = 3,
        backoff: float = 2.0,
        session_factory: Callable[[], Any] | None = None,
    ):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.timeout = timeout
        self._session = (session_factory or (lambda: make_session(retries, backoff, USER_AGENT)))()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Trả về ma trận float32 (len(texts), dim) đã chuẩn hóa L2, đúng thứ tự texts."""
        try:
            resp = self._session.request("POST", self.url, json={"model": self.model, "input": texts},
                                         timeout=self.timeout)
        except requests.Timeout as e:
            raise EmbeddingError("timeout", str(e)) from e
        except requests.ConnectionError as e:
            raise EmbeddingError("connection", str(e)) from e
        except requests.RequestException as e:
            raise EmbeddingError("request", str(e)) from e

        if resp.status_code != 200:
            message = _error_message(resp)
            error = TextTooLong if any(m in message.lower() for m in CONTEXT_ERROR_MARKERS) else EmbeddingError
            raise error(f"http_{resp.status_code}", message[:200])
        try:
            body = resp.json()
        except ValueError as e:
            raise EmbeddingError("bad_json", str(e)) from e
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingError("bad_response", f"cần {len(texts)} vector, nhận {len(data) if isinstance(data, list) else data!r}")
        data = sorted(data, key=lambda x: x.get("index", 0))
        return l2_normalize(np.asarray([x["embedding"] for x in data], dtype=np.float32))


@dataclass
class EmbedReport:
    total_chunks: int = 0
    already_indexed: int = 0
    previously_skipped: int = 0
    embedded: int = 0
    skipped_too_long: int = 0
    vectors: int = 0
    dim: int | None = None
    remaining: int = 0


def _embed_batch(client: EmbeddingClient, ids: list[int], texts: list[str],
                 skipped: dict[str, str], report: EmbedReport) -> tuple[np.ndarray | None, list[int]]:
    try:
        return client.embed(texts), list(ids)
    except TextTooLong as e:
        if len(texts) == 1:
            skipped[str(ids[0])] = e.detail or e.reason
            report.skipped_too_long += 1
            log.warning("Bỏ qua chunk %s (%s ký tự): vượt context model", ids[0], len(texts[0]))
            return None, []
    # Một text quá dài làm hỏng cả batch → embed lẻ từng text
    vecs, ok = [], []
    for faiss_id, text in zip(ids, texts):
        vec, got = _embed_batch(client, [faiss_id], [text], skipped, report)
        if got:
            vecs.append(vec[0])
            ok.extend(got)
    return (np.stack(vecs) if vecs else None), ok


def run_embed(
    client: EmbeddingClient,
    db_path: Path,
    index_path: Path,
    skipped_path: Path,
    batch_size: int = 8,
    max_chunks: int | None = None,
    save_every: float = 300.0,
    clock: Callable[[], float] = time.monotonic,
) -> EmbedReport:
    """Embed các chunk chưa có trong index. Lưu index mỗi `save_every` giây và khi dừng (kể cả lỗi/Ctrl+C).

    Lỗi API khác "vượt context" làm dừng lần chạy thay vì bỏ qua chunk: code cũ bỏ qua mọi lỗi,
    nên khi server embedding sập thì phần còn lại bị đánh dấu skip hàng loạt.
    """
    db_path, index_path, skipped_path = Path(db_path), Path(index_path), Path(skipped_path)
    report = EmbedReport(total_chunks=count_chunks(db_path))

    index = read_index(index_path) if index_path.exists() else None
    done: set[int] = set()
    if index is not None:
        ids = indexed_ids(index)
        if len(ids) and (ids.min() < 0 or ids.max() >= report.total_chunks):
            raise IndexMismatch(
                f"{index_path} có faiss_id {ids.max()} nhưng chunks.db chỉ có {report.total_chunks} chunk: "
                "index được dựng từ chunks.db khác. Xoá index rồi embed lại."
            )
        done = set(ids.tolist())
    skipped: dict[str, str] = json.loads(skipped_path.read_text(encoding="utf-8")) if skipped_path.exists() else {}
    report.already_indexed, report.previously_skipped = len(done), len(skipped)

    pending_ids: list[int] = []
    pending_texts: list[str] = []
    dirty = False
    last_save = started = clock()
    next_log = LOG_EVERY

    def save() -> None:
        nonlocal dirty, last_save
        if index is not None:
            write_index_atomic(index, index_path)
        atomic_write_json(skipped_path, skipped)
        dirty, last_save = False, clock()
        log.info("Đã lưu index: %s vector", index.ntotal if index is not None else 0)

    def flush_pending() -> None:
        nonlocal index, dirty, next_log
        vecs, ids = _embed_batch(client, pending_ids, pending_texts, skipped, report)
        pending_ids.clear()
        pending_texts.clear()
        dirty = True
        if not ids:
            return
        if index is None:
            index = new_flat_index(vecs.shape[1])
        elif vecs.shape[1] != index.d:
            raise IndexMismatch(f"model trả vector {vecs.shape[1]} chiều, index hiện có {index.d} chiều")
        index.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))
        report.embedded += len(ids)
        if report.embedded >= next_log:
            rate = report.embedded / max(clock() - started, 1e-9)
            log.info("+%s chunk (%.1f chunk/s), index %s/%s", report.embedded, rate, index.ntotal, report.total_chunks)
            next_log += LOG_EVERY

    chunks = iter_chunk_texts(db_path)
    try:
        for faiss_id, text in chunks:
            if faiss_id in done or str(faiss_id) in skipped:
                continue
            if max_chunks is not None and report.embedded + report.skipped_too_long + len(pending_ids) >= max_chunks:
                break
            pending_ids.append(faiss_id)
            pending_texts.append(text)
            if len(pending_ids) >= batch_size:
                flush_pending()
                if clock() - last_save >= save_every:
                    save()
        if pending_ids:
            flush_pending()
    finally:
        chunks.close()
        if dirty:
            save()

    report.vectors = index.ntotal if index is not None else 0
    report.dim = index.d if index is not None else None
    report.remaining = report.total_chunks - report.vectors - len(skipped)
    return report
