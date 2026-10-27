import json

import faiss
import numpy as np
import pytest
import requests

from legal_rag.chunk_db import ChunkDbWriter
from legal_rag.embedding import EmbeddingClient, EmbeddingError, TextTooLong, run_embed
from legal_rag.vector_index import IndexMismatch, indexed_ids, read_index

DIM = 8
TOO_LONG = "QUÁ DÀI"


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("Expecting value")
        return self._body


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_client(*responses, **kwargs):
    session = FakeSession(*responses)
    return EmbeddingClient("http://lm:1234/v1/", "m", session_factory=lambda: session, **kwargs), session


def test_client_sends_batch_and_returns_normalized_vectors_in_order():
    body = {"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 4]}]}
    client, session = make_client(FakeResponse(200, body), api_key="k")
    vecs = client.embed(["a", "b"])

    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "http://lm:1234/v1/embeddings")
    assert kwargs["json"] == {"model": "m", "input": ["a", "b"]}
    assert session.headers["Authorization"] == "Bearer k"
    np.testing.assert_allclose(vecs, [[0.6, 0.8], [0, 1]])
    assert vecs.dtype == np.float32


def test_context_length_error_is_text_too_long_whatever_the_status():
    client, _ = make_client(
        FakeResponse(400, {"error": {"message": "This model's maximum context length is 8192 tokens"}}),
        FakeResponse(500, None, text="input_tokens exceeds limit"),
        FakeResponse(500, {"error": "model crashed"}),
    )
    with pytest.raises(TextTooLong):
        client.embed(["x"])
    with pytest.raises(TextTooLong):
        client.embed(["x"])
    with pytest.raises(EmbeddingError) as exc:
        client.embed(["x"])
    assert not isinstance(exc.value, TextTooLong)
    assert exc.value.reason == "http_500"


def test_transport_and_shape_errors_are_mapped():
    client, _ = make_client(requests.ConnectionError("c"), FakeResponse(200, {"data": [{"embedding": [1]}]}))
    with pytest.raises(EmbeddingError) as exc:
        client.embed(["a"])
    assert exc.value.reason == "connection"
    with pytest.raises(EmbeddingError) as exc:
        client.embed(["a", "b"])
    assert exc.value.reason == "bad_response"


class FakeEmbedder:
    """Vector tất định theo nội dung; text chứa TOO_LONG làm hỏng cả request như server thật."""

    def __init__(self, dim=DIM, fail_on_call=None):
        self.dim = dim
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise EmbeddingError("http_503")
        if any(TOO_LONG in t for t in texts):
            raise TextTooLong("http_400", "maximum context length")
        rng = [np.random.default_rng(abs(hash(t)) % 2**32) for t in texts]
        vecs = np.stack([r.standard_normal(self.dim) for r in rng]).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def index_dir(tmp_path):
    with ChunkDbWriter(tmp_path / "chunks.db") as db:
        db.add_doc("D", "1/2020/QH14", "Luật")
        for i in range(10):
            db.add_chunk(f"c{i}", "D", "1/2020/QH14", "Luật", None, None, TOO_LONG if i == 4 else f"chunk {i}")
    return tmp_path


def embed(index_dir, client, **kwargs):
    return run_embed(client, index_dir / "chunks.db", index_dir / "faiss.index", index_dir / "embed_skipped.json",
                     batch_size=3, **kwargs)


def test_full_run_indexes_every_chunk_except_too_long(index_dir):
    client = FakeEmbedder()
    report = embed(index_dir, client)

    index = read_index(index_dir / "faiss.index")
    assert sorted(indexed_ids(index).tolist()) == [0, 1, 2, 3, 5, 6, 7, 8, 9]
    assert (report.embedded, report.skipped_too_long, report.vectors, report.dim, report.remaining) == (9, 1, 9, DIM, 0)
    assert list(json.loads((index_dir / "embed_skipped.json").read_text(encoding="utf-8"))) == ["4"]
    # Batch [3,4,5] hỏng vì chunk 4 → embed lẻ từng chunk
    assert client.calls[1] == ["chunk 3", TOO_LONG, "chunk 5"]
    assert client.calls[2:5] == [["chunk 3"], [TOO_LONG], ["chunk 5"]]

    # Vector lưu đúng faiss_id: tìm chính nó ra top 1
    q = FakeEmbedder().embed(["chunk 7"])
    assert index.search(q, 1)[1][0][0] == 7


def test_resume_skips_indexed_and_skipped_chunks(index_dir):
    first = embed(index_dir, FakeEmbedder(), max_chunks=4)
    assert (first.embedded, first.skipped_too_long, first.remaining) == (4, 0, 6)
    assert indexed_ids(read_index(index_dir / "faiss.index")).tolist() == [0, 1, 2, 3]

    client = FakeEmbedder()
    second = embed(index_dir, client)
    assert (second.already_indexed, second.embedded, second.skipped_too_long, second.vectors) == (4, 5, 1, 9)
    assert client.calls[0] == [TOO_LONG, "chunk 5", "chunk 6"]

    third_client = FakeEmbedder()
    third = embed(index_dir, third_client)
    assert (third.embedded, third.previously_skipped, third_client.calls) == (0, 1, [])


def test_api_error_stops_run_but_keeps_progress(index_dir):
    # call 1 = [0,1,2] ok, call 2 = [3,4,5] vượt context, call 3 = [3] lỗi server
    with pytest.raises(EmbeddingError):
        embed(index_dir, FakeEmbedder(fail_on_call=3))
    assert indexed_ids(read_index(index_dir / "faiss.index")).tolist() == [0, 1, 2]
    assert json.loads((index_dir / "embed_skipped.json").read_text(encoding="utf-8")) == {}

    report = embed(index_dir, FakeEmbedder())
    assert (report.already_indexed, report.vectors) == (3, 9)


def test_index_from_other_chunk_set_is_rejected(index_dir):
    index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
    index.add_with_ids(np.ones((1, DIM), dtype=np.float32), np.array([99], dtype=np.int64))
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with pytest.raises(IndexMismatch):
        embed(index_dir, FakeEmbedder())


def test_dimension_change_is_rejected(index_dir):
    embed(index_dir, FakeEmbedder(), max_chunks=3)
    with pytest.raises(IndexMismatch):
        embed(index_dir, FakeEmbedder(dim=DIM * 2))
