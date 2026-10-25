import json

import pytest

from legal_rag.config import LIST_QUERY
from legal_rag.storage import (
    Checkpoint,
    CheckpointMismatch,
    JsonlStore,
    atomic_write_json,
    repair_trailing_partial_line,
)


def test_atomic_write_json_replaces_file_and_leaves_no_tmp(tmp_path):
    target = tmp_path / "sub" / "ck.json"
    atomic_write_json(target, {"a": 1})
    atomic_write_json(target, {"a": 2, "text": "Điều"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2, "text": "Điều"}
    assert list(target.parent.iterdir()) == [target]


def test_repair_cuts_partial_last_line(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_bytes(b'{"doc_id": "1"}\n{"doc_id": "2", "detail": {"x')
    cut = repair_trailing_partial_line(path)
    assert cut == len(b'{"doc_id": "2", "detail": {"x')
    assert path.read_bytes() == b'{"doc_id": "1"}\n'


def test_repair_is_noop_for_clean_or_empty_file(tmp_path):
    clean = tmp_path / "clean.jsonl"
    clean.write_bytes(b'{"doc_id": "1"}\n')
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert repair_trailing_partial_line(clean) == 0
    assert repair_trailing_partial_line(empty) == 0
    assert repair_trailing_partial_line(tmp_path / "missing.jsonl") == 0


def test_repair_single_partial_line_empties_file(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_bytes(b'{"doc_id": "1", "det')
    repair_trailing_partial_line(path)
    assert path.read_bytes() == b""


def test_store_writes_doc_id_first_and_reloads_seen_ids(tmp_path):
    path = tmp_path / "docs.jsonl"
    with JsonlStore(path) as store:
        store.append({"detail": {"title": "Luật"}, "doc_id": 42})
        store.append({"doc_id": 'id "có" ngoặc'})
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith('{"doc_id": "42"')
    with JsonlStore(path) as store:
        assert "42" in store
        assert 'id "có" ngoặc' in store
        assert len(store) == 2


def test_store_recovers_from_crash_mid_write(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_bytes(b'{"doc_id": "1"}\n{"doc_id": "2", "detail": {"x')
    with JsonlStore(path) as store:
        assert store.repaired_bytes > 0
        assert "1" in store and "2" not in store
        store.append({"doc_id": "2"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["doc_id"] for line in lines] == ["1", "2"]


def test_store_uses_lf_line_endings(tmp_path):
    path = tmp_path / "docs.jsonl"
    with JsonlStore(path) as store:
        store.append({"doc_id": "1"})
    assert b"\r\n" not in path.read_bytes()


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "ck.json"
    ck = Checkpoint.load(path, LIST_QUERY, page_size=10)
    ck.next_page = 7
    ck.update_total(37673)
    ck.failed_docs["123"] = "timeout"
    ck.failed_pages["5"] = "http_503"
    ck.save()

    again = Checkpoint.load(path, LIST_QUERY, page_size=10)
    assert again.next_page == 7
    assert again.total == 37673
    assert again.last_page == 3768
    assert again.failed_docs == {"123": "timeout"}
    assert again.failed_pages == {"5": "http_503"}


def test_checkpoint_refuses_different_page_size_or_query(tmp_path):
    path = tmp_path / "ck.json"
    Checkpoint.load(path, LIST_QUERY, page_size=10).save()
    with pytest.raises(CheckpointMismatch):
        Checkpoint.load(path, LIST_QUERY, page_size=100)
    with pytest.raises(CheckpointMismatch):
        Checkpoint.load(path, {**LIST_QUERY, "sortByViewCount": False}, page_size=10)
    fresh = Checkpoint.load(path, LIST_QUERY, page_size=100, reset=True)
    assert fresh.next_page == 1 and fresh.page_size == 100
