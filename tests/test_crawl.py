import json

import pytest

from legal_rag.crawl import CrawlAborted, CrawlOptions, crawl, reconcile, retry_failed
from legal_rag.storage import Checkpoint
from tests.conftest import FakeClient, open_ckpt, open_store, read_doc_ids


def run(client, raw_dir, **opts):
    with open_store(raw_dir) as store:
        ckpt = open_ckpt(raw_dir, opts.get("page_size", 10))
        stats = crawl(client, store, ckpt, CrawlOptions(**opts))
    return stats, ckpt


def test_respects_max_pages(client, raw_dir):
    stats, ckpt = run(client, raw_dir, max_pages=2)
    assert stats.fetched == 20
    assert stats.stop_reason == "max_pages"
    assert ckpt.next_page == 3
    assert [p for p, _ in client.list_calls] == [1, 2]
    record = json.loads((raw_dir / "docs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(record) == {"doc_id", "page", "fetched_at", "list_item", "detail"}
    assert record["page"] == 1


def test_stops_at_end_using_api_total(client, raw_dir):
    stats, ckpt = run(client, raw_dir, max_pages=None)
    assert stats.fetched == 25
    assert stats.stop_reason == "end"
    assert ckpt.done and ckpt.last_page == 3
    # Không gọi trang 4 khi đã biết total = 25 (code cũ gọi tới trang 36.916)
    assert [p for p, _ in client.list_calls] == [1, 2, 3]


def test_resume_continues_without_duplicates(client, raw_dir):
    run(client, raw_dir, max_pages=1)
    stats, ckpt = run(client, raw_dir, max_pages=None)
    ids = read_doc_ids(raw_dir / "docs.jsonl")
    assert len(ids) == len(set(ids)) == 25
    assert len(client.detail_calls) == 25
    assert stats.fetched == 15


def test_rerunning_a_page_skips_saved_docs(client, raw_dir):
    run(client, raw_dir, max_pages=1)
    ckpt = open_ckpt(raw_dir)
    ckpt.next_page = 1  # giả lập crash trước khi checkpoint kịp lưu
    ckpt.save()
    stats, _ = run(client, raw_dir, max_pages=1)
    assert stats.already_seen == 10
    assert stats.fetched == 0
    assert len(client.detail_calls) == 10


def test_max_docs_stops_mid_page_and_resume_finishes_page(client, raw_dir):
    stats, ckpt = run(client, raw_dir, max_pages=None, max_docs=5)
    assert stats.fetched == 5
    assert stats.stop_reason == "max_docs"
    assert ckpt.next_page == 1
    stats, ckpt = run(client, raw_dir, max_pages=1)
    assert stats.fetched == 5 and stats.already_seen == 5
    assert ckpt.next_page == 2


def test_failed_page_is_recorded_then_retried(client, raw_dir):
    client.page_errors[2] = "http_503"
    _, ckpt = run(client, raw_dir, max_pages=3)
    assert ckpt.failed_pages == {"2": "http_503"}
    assert len(read_doc_ids(raw_dir / "docs.jsonl")) == 15

    client.page_errors.clear()
    with open_store(raw_dir) as store:
        ckpt = open_ckpt(raw_dir)
        retry_failed(client, store, ckpt, CrawlOptions())
    assert ckpt.failed_pages == {}
    assert len(read_doc_ids(raw_dir / "docs.jsonl")) == 25


def test_failed_doc_is_recorded_then_retried(client, raw_dir):
    client.doc_errors["1003"] = "timeout"
    _, ckpt = run(client, raw_dir, max_pages=1)
    assert ckpt.failed_docs == {"1003": "timeout"}
    assert "1003" not in read_doc_ids(raw_dir / "docs.jsonl")

    client.doc_errors.clear()
    with open_store(raw_dir) as store:
        ckpt = open_ckpt(raw_dir)
        stats = retry_failed(client, store, ckpt, CrawlOptions())
    assert stats.fetched == 1
    assert ckpt.failed_docs == {}
    assert "1003" in read_doc_ids(raw_dir / "docs.jsonl")


def test_circuit_breaker_stops_and_keeps_page_for_redo(client, raw_dir):
    for item in client.items[:10]:
        client.doc_errors[item["id"]] = "http_403"
    with pytest.raises(CrawlAborted, match="10/10"):
        run(client, raw_dir, max_pages=3)
    saved = Checkpoint.read(raw_dir / "checkpoint.json")
    assert saved["next_page"] == 1
    assert len(saved["failed_docs"]) == 10
    assert [p for p, _ in client.list_calls] == [1]  # không chạy tiếp sang trang 2


def test_consecutive_page_errors_abort(client, raw_dir):
    for page in range(1, 10):
        client.page_errors[page] = "connection"
    with pytest.raises(CrawlAborted, match="liên tiếp"):
        run(client, raw_dir, max_pages=None)
    assert len(Checkpoint.read(raw_dir / "checkpoint.json")["failed_pages"]) == 5


def test_skip_invalid_status_avoids_detail_calls(raw_dir):
    client = FakeClient(n_docs=10, status_for=lambda i: "Hết hiệu lực toàn bộ" if i % 2 else "Còn hiệu lực")
    stats, _ = run(client, raw_dir, max_pages=1, skip_invalid_status=True)
    assert stats.skipped_status == 5
    assert stats.fetched == 5
    assert len(client.detail_calls) == 5


def test_workers_keep_catalog_order(client, raw_dir):
    run(client, raw_dir, max_pages=None, workers=4)
    assert read_doc_ids(raw_dir / "docs.jsonl") == [it["id"] for it in client.items]


def test_reconcile_fetches_docs_missed_by_pagination(client, raw_dir):
    run(client, raw_dir, max_pages=1)
    with open_store(raw_dir) as store:
        ckpt = open_ckpt(raw_dir)
        stats = reconcile(client, store, ckpt, page_size=100)
    assert stats.complete
    assert stats.listed == 25 and stats.missing == 15 and stats.fetched == 15
    assert stats.extra_in_store == 0
    assert len(set(read_doc_ids(raw_dir / "docs.jsonl"))) == 25
