import faiss
import numpy as np
import pytest

from legal_rag.vector_index import IndexMismatch, convert_fp16, indexed_ids, new_flat_index, read_index


def write_flat(path, n, d=32, seed=0):
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n, d)).astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    index = new_flat_index(d)
    ids = np.arange(n, dtype=np.int64)[::-1].copy()  # id khác vị trí, để kiểm tra id được giữ
    index.add_with_ids(xb, ids)
    faiss.write_index(index, str(path))
    return xb, ids


def test_fp16_keeps_ids_and_recall(tmp_path):
    xb, ids = write_flat(tmp_path / "faiss.index", 500)
    report = convert_fp16(tmp_path / "faiss.index", tmp_path / "faiss_fp16.index")

    dst = read_index(tmp_path / "faiss_fp16.index")
    assert isinstance(faiss.downcast_index(dst.index), faiss.IndexScalarQuantizer)
    assert indexed_ids(dst).tolist() == ids.tolist()
    assert dst.search(xb[:1], 1)[1][0][0] == ids[0]
    assert report.vectors == 500 and report.recall >= 0.95
    assert not (tmp_path / "faiss_fp16.index.tmp").exists()


def test_fp16_small_index(tmp_path):
    write_flat(tmp_path / "faiss.index", 3)
    report = convert_fp16(tmp_path / "faiss.index", tmp_path / "faiss_fp16.index")
    assert (report.queries, report.top_k, report.recall) == (3, 3, 1.0)


def test_fp16_rejects_empty_or_non_flat_index(tmp_path):
    faiss.write_index(new_flat_index(8), str(tmp_path / "empty.index"))
    with pytest.raises(IndexMismatch):
        convert_fp16(tmp_path / "empty.index", tmp_path / "out.index")

    faiss.write_index(faiss.IndexFlatIP(8), str(tmp_path / "plain.index"))
    with pytest.raises(IndexMismatch):
        convert_fp16(tmp_path / "plain.index", tmp_path / "out.index")
