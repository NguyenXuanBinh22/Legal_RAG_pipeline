"""FAISS index: tạo/đọc/ghi nguyên tử, và nén fp16 cho máy ít RAM.

faiss.index (float32, IndexIDMap(IndexFlatIP)) là bản gốc mà Phase 4 ghi thêm và resume.
faiss_fp16.index là bản dẫn xuất (~1/2 RAM) mà SearchService ưu tiên đọc, dựng lại sau mỗi lần embed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

ADD_BATCH_SIZE = 50_000


class IndexMismatch(Exception):
    """Index không khớp với chunks.db hiện tại (dim khác, hoặc có id ngoài phạm vi)."""


def new_flat_index(dim: int) -> faiss.Index:
    # Vector đã chuẩn hóa L2 → inner product = cosine (SearchService cũng chuẩn hóa query)
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))


def read_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def write_index_atomic(index: faiss.Index, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def indexed_ids(index: faiss.Index) -> np.ndarray:
    return faiss.vector_to_array(index.id_map)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


@dataclass
class Fp16Report:
    vectors: int
    dim: int
    recall: float
    top_k: int
    queries: int
    latency_ms_float32: float
    latency_ms_fp16: float
    size_mb: float


def convert_fp16(src_path: Path, dst_path: Path, verify_queries: int = 200, top_k: int = 10,
                 seed: int = 0) -> Fp16Report:
    """IndexIDMap(IndexFlatIP) float32 → IndexIDMap(IndexScalarQuantizer fp16), kèm kiểm tra recall."""
    src = read_index(src_path)
    flat = faiss.downcast_index(src.index) if isinstance(src, faiss.IndexIDMap) else None
    if not isinstance(flat, faiss.IndexFlat):
        raise IndexMismatch(f"{src_path} không phải IndexIDMap(IndexFlat): {type(src).__name__}")
    n, d = src.ntotal, src.d
    if n == 0:
        raise IndexMismatch(f"{src_path} rỗng, chạy embed trước")

    ids = indexed_ids(src)
    # View trực tiếp vào bộ nhớ của index gốc, không copy thêm (~3.2GB với dữ liệu đầy đủ)
    xb = faiss.rev_swig_ptr(flat.get_xb(), n * d).reshape(n, d)

    dst = faiss.IndexIDMap(faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT))
    for start in range(0, n, ADD_BATCH_SIZE):
        dst.add_with_ids(xb[start:start + ADD_BATCH_SIZE], ids[start:start + ADD_BATCH_SIZE])

    k = min(top_k, n)
    rng = np.random.default_rng(seed)
    queries = np.ascontiguousarray(xb[rng.choice(n, min(verify_queries, n), replace=False)])
    _, gt = src.search(queries, k)
    _, res = dst.search(queries, k)
    recall = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(gt, res)]))

    # Latency 1 query/lần, đúng cách SearchService gọi
    latency = {}
    for name, idx in (("float32", src), ("fp16", dst)):
        sample = queries[:20]
        t = time.perf_counter()
        for q in sample:
            idx.search(q[None, :], k)
        latency[name] = (time.perf_counter() - t) / len(sample) * 1000

    write_index_atomic(dst, dst_path)
    return Fp16Report(n, d, recall, k, len(queries), latency["float32"], latency["fp16"],
                      os.path.getsize(dst_path) / 2**20)
