"""Phase 1: crawl danh sách + chi tiết văn bản, resume an toàn, không ghi trùng."""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import KEEP_STATUSES
from .storage import Checkpoint, JsonlStore, now_iso
from .vbpl_client import VbplError

log = logging.getLogger(__name__)

_KEEP = {unicodedata.normalize("NFC", s) for s in KEEP_STATUSES}


class CrawlAborted(RuntimeError):
    """Dừng sớm khi phần lớn văn bản trong 1 trang lỗi (API đổi, bị chặn, mất mạng)."""


@dataclass
class CrawlOptions:
    page_size: int = 10
    max_pages: int | None = 2
    max_docs: int | None = None
    workers: int = 1
    skip_invalid_status: bool = False
    breaker_min_docs: int = 3
    breaker_ratio: float = 0.8
    max_consecutive_page_errors: int = 5


@dataclass
class CrawlStats:
    pages: int = 0
    listed: int = 0
    attempted: int = 0
    fetched: int = 0
    already_seen: int = 0
    skipped_status: int = 0
    failed_docs: int = 0
    failed_pages: int = 0
    reasons: Counter = field(default_factory=Counter)
    stop_reason: str = ""


@dataclass
class ReconcileStats:
    listed: int = 0
    list_pages: int = 0
    list_errors: int = 0
    missing: int = 0
    fetched: int = 0
    failed: int = 0
    complete: bool = False
    extra_in_store: int | None = None


def doc_id_of(item: dict) -> str | None:
    value = item.get("id")
    return str(value) if value not in (None, "") else None


def status_name(obj: dict | None) -> str | None:
    eff = obj.get("effStatus") if isinstance(obj, dict) else None
    if isinstance(eff, dict) and eff.get("name"):
        return unicodedata.normalize("NFC", eff["name"])
    return None


def make_record(doc_id: str, page: int | None, list_item: dict | None, detail: dict) -> dict:
    return {"doc_id": doc_id, "page": page, "fetched_at": now_iso(), "list_item": list_item, "detail": detail}


def _fetch_many(client, jobs: list[tuple[str, dict | None]], workers: int):
    """Gọi detail cho nhiều văn bản. Kết quả giữ đúng thứ tự `jobs` để ghi file tất định."""

    def one(job):
        try:
            return job, client.get_detail(job[0]), None
        except VbplError as e:
            return job, None, e

    if workers <= 1:
        return [one(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def _store_results(results, store: JsonlStore, ckpt: Checkpoint, stats: CrawlStats, page: int | None) -> Counter:
    failures: Counter = Counter()
    for (doc_id, item), detail, err in results:
        stats.attempted += 1
        if err is not None:
            ckpt.failed_docs[doc_id] = err.reason
            failures[err.reason] += 1
            stats.failed_docs += 1
            stats.reasons[f"doc:{err.reason}"] += 1
            log.warning("Văn bản %s lỗi: %s", doc_id, err)
        else:
            store.append(make_record(doc_id, page, item, detail))
            ckpt.failed_docs.pop(doc_id, None)
            stats.fetched += 1
    store.flush()
    return failures


def _process_page(client, store, ckpt, opts: CrawlOptions, stats: CrawlStats, page: int, items: list[dict],
                  limit_docs: bool = True) -> bool:
    """Xử lý 1 trang. Trả về False nếu bị cắt giữa trang vì max_docs (resume sẽ làm lại trang này)."""
    jobs: list[tuple[str, dict]] = []
    for item in items:
        stats.listed += 1
        doc_id = doc_id_of(item)
        if doc_id is None:
            stats.reasons["item_without_id"] += 1
            continue
        if doc_id in store:
            stats.already_seen += 1
            continue
        if opts.skip_invalid_status and status_name(item) not in _KEEP:
            stats.skipped_status += 1
            continue
        jobs.append((doc_id, item))

    truncated = False
    if limit_docs and opts.max_docs is not None:
        room = max(opts.max_docs - stats.attempted, 0)
        if len(jobs) > room:
            jobs, truncated = jobs[:room], True

    results = _fetch_many(client, jobs, opts.workers)
    failures = _store_results(results, store, ckpt, stats, page)

    n_fail = sum(failures.values())
    if len(results) >= opts.breaker_min_docs and n_fail / len(results) >= opts.breaker_ratio:
        ckpt.next_page = page  # làm lại trang này sau khi sửa nguyên nhân
        ckpt.save()
        raise CrawlAborted(
            f"Trang {page}: {n_fail}/{len(results)} văn bản lỗi {dict(failures)}. "
            "Dừng để không chạy tiếp trên API đang hỏng. Kiểm tra mạng/API rồi chạy lại lệnh crawl."
        )
    return not truncated


def crawl(client, store: JsonlStore, ckpt: Checkpoint, opts: CrawlOptions) -> CrawlStats:
    stats = CrawlStats()
    page = ckpt.next_page
    consecutive_page_errors = 0

    while True:
        if opts.max_pages is not None and stats.pages >= opts.max_pages:
            stats.stop_reason = "max_pages"
            break
        if opts.max_docs is not None and stats.attempted >= opts.max_docs:
            stats.stop_reason = "max_docs"
            break
        if ckpt.last_page is not None and page > ckpt.last_page:
            ckpt.done = True
            stats.stop_reason = "end"
            break

        try:
            listing = client.list_page(page, opts.page_size)
        except VbplError as e:
            log.warning("Trang %s lỗi: %s", page, e)
            ckpt.failed_pages[str(page)] = e.reason
            stats.failed_pages += 1
            stats.reasons[f"page:{e.reason}"] += 1
            stats.pages += 1
            page += 1
            ckpt.next_page = page
            ckpt.save()
            consecutive_page_errors += 1
            if consecutive_page_errors >= opts.max_consecutive_page_errors:
                raise CrawlAborted(
                    f"{consecutive_page_errors} trang liên tiếp lỗi (gần nhất: {e.reason}). "
                    "Các trang này đã ghi vào failed_pages; sửa nguyên nhân rồi chạy crawl --retry-failed."
                )
            continue

        consecutive_page_errors = 0
        ckpt.update_total(listing.total)
        if not listing.items:
            ckpt.done = True
            ckpt.next_page = page
            stats.stop_reason = "end"
            break

        finished = _process_page(client, store, ckpt, opts, stats, page, listing.items)
        ckpt.failed_pages.pop(str(page), None)
        stats.pages += 1
        if finished:
            page += 1
        ckpt.next_page = page
        ckpt.save()
        log.info("Trang %s/%s xong | đã lưu %s văn bản | lỗi %s",
                 listing.page_number, ckpt.last_page or "?", len(store), len(ckpt.failed_docs))
        if not finished:
            stats.stop_reason = "max_docs"
            break

    ckpt.save()
    return stats


def retry_failed(client, store: JsonlStore, ckpt: Checkpoint, opts: CrawlOptions) -> CrawlStats:
    """Thử lại các trang và văn bản lỗi đã ghi trong checkpoint."""
    stats = CrawlStats()

    for page_key in sorted(ckpt.failed_pages, key=int):
        page = int(page_key)
        try:
            listing = client.list_page(page, opts.page_size)
        except VbplError as e:
            ckpt.failed_pages[page_key] = e.reason
            stats.failed_pages += 1
            log.warning("Trang %s vẫn lỗi: %s", page, e)
            continue
        _process_page(client, store, ckpt, opts, stats, page, listing.items, limit_docs=False)
        ckpt.failed_pages.pop(page_key, None)
        stats.pages += 1
        ckpt.save()

    for doc_id in [d for d in ckpt.failed_docs if d in store]:
        ckpt.failed_docs.pop(doc_id)  # đã có từ lượt khác (vd. reconcile)
    jobs = [(doc_id, None) for doc_id in ckpt.failed_docs]
    _store_results(_fetch_many(client, jobs, opts.workers), store, ckpt, stats, page=None)
    ckpt.save()
    stats.stop_reason = "retry_done"
    return stats


def reconcile(client, store: JsonlStore, ckpt: Checkpoint, page_size: int = 100,
              max_pages: int | None = None, workers: int = 1,
              max_consecutive_errors: int = 5) -> ReconcileStats:
    """Liệt kê lại toàn bộ danh sách (page lớn, tốn ít request) và crawl văn bản còn thiếu.

    Cần thiết vì danh sách sắp theo lượt xem: văn bản có thể trượt giữa các trang trong lúc crawl.
    """
    stats = ReconcileStats()
    listed: dict[str, dict] = {}
    page = 1
    consecutive_errors = 0
    while max_pages is None or stats.list_pages < max_pages:
        try:
            listing = client.list_page(page, page_size)
        except VbplError as e:
            stats.list_errors += 1
            log.warning("Reconcile: trang %s lỗi: %s", page, e)
            page += 1
            stats.list_pages += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                raise CrawlAborted(f"Reconcile: {consecutive_errors} trang liên tiếp lỗi (gần nhất: {e.reason}).")
            continue
        consecutive_errors = 0
        stats.list_pages += 1
        if not listing.items:
            stats.complete = stats.list_errors == 0
            break
        for item in listing.items:
            doc_id = doc_id_of(item)
            if doc_id is not None:
                listed.setdefault(doc_id, item)
        if listing.total is not None and page * page_size >= listing.total:
            stats.complete = stats.list_errors == 0
            break
        page += 1

    stats.listed = len(listed)
    missing = [(doc_id, item) for doc_id, item in listed.items() if doc_id not in store]
    stats.missing = len(missing)
    crawl_stats = CrawlStats()
    _store_results(_fetch_many(client, missing, workers), store, ckpt, crawl_stats, page=None)
    stats.fetched, stats.failed = crawl_stats.fetched, crawl_stats.failed_docs
    if stats.complete:
        stats.extra_in_store = len(store.seen - listed.keys())
    ckpt.save()
    return stats
