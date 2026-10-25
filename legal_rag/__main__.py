"""CLI: python -m legal_rag {probe|crawl|reconcile|status|preprocess}"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from .config import LIST_QUERY, Paths
from .crawl import CrawlAborted, CrawlOptions, crawl, reconcile, retry_failed, status_name
from .preprocess import run_preprocess
from .storage import Checkpoint, CheckpointMismatch, JsonlStore
from .structure import build_provision_tree
from .vbpl_client import VbplClient, VbplError

log = logging.getLogger("legal_rag")


def _setup_logging(verbose: bool, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _print_kv(title: str, pairs: dict) -> None:
    print(f"\n== {title} ==")
    width = max((len(k) for k in pairs), default=0)
    for key, value in pairs.items():
        print(f"  {key:<{width}}  {value}")


def cmd_probe(args) -> int:
    client = VbplClient(delay=0)
    listing = client.list_page(1, args.n)
    page_size_10 = -(-listing.total // 10) if listing.total else "?"
    _print_kv("Danh sách", {
        "total (số văn bản)": listing.total,
        "số trang (size 10)": page_size_10,
        "items trang 1": len(listing.items),
    })
    for item in listing.items:
        print(f"  - {item.get('id')} | {item.get('docNum')} | {status_name(item)}")

    first = str(listing.items[0]["id"])
    detail = client.get_detail(first)
    html = (detail.get("documentContent") or {}).get("content") or ""
    tree, stats = build_provision_tree(html)
    _print_kv(f"Chi tiết văn bản {first}", {
        "docNum": detail.get("docNum"),
        "effStatus": status_name(detail),
        "độ dài HTML": len(html),
        "node theo cấp": dict(stats.levels),
        "node gốc của cây": len(tree),
        "node thiếu id": stats.skipped_no_id,
        "class prov-* lạ": dict(stats.unknown_classes) or "-",
    })
    print("\nProbe không ghi file nào.")
    return 0


def cmd_crawl(args) -> int:
    paths = Paths(Path(args.data_dir))
    _setup_logging(args.verbose, paths.log)
    opts = CrawlOptions(
        page_size=args.page_size,
        max_pages=None if args.all else args.max_pages,
        max_docs=args.max_docs,
        workers=args.workers,
        skip_invalid_status=args.skip_invalid_status,
    )
    try:
        ckpt = Checkpoint.load(paths.checkpoint, LIST_QUERY, args.page_size, reset=args.reset)
    except CheckpointMismatch as e:
        log.error("%s", e)
        return 2

    client = VbplClient(delay=args.delay)
    with JsonlStore(paths.docs) as store:
        if store.repaired_bytes:
            log.warning("Đã cắt %s byte dòng ghi dở ở cuối %s", store.repaired_bytes, paths.docs)
        log.info("Bắt đầu: %s văn bản đã có, trang tiếp theo %s", len(store), ckpt.next_page)
        try:
            stats = retry_failed(client, store, ckpt, opts) if args.retry_failed else crawl(client, store, ckpt, opts)
        except CrawlAborted as e:
            log.error("DỪNG: %s", e)
            return 2
        total_docs = len(store)

    _print_kv("Kết quả crawl", {
        "trang đã xử lý": stats.pages,
        "văn bản mới": stats.fetched,
        "đã có sẵn (bỏ qua)": stats.already_seen,
        "bỏ qua do hiệu lực": stats.skipped_status,
        "văn bản lỗi": stats.failed_docs,
        "trang lỗi": stats.failed_pages,
        "lý do lỗi": dict(stats.reasons) or "-",
        "dừng vì": stats.stop_reason,
        "tổng trong docs.jsonl": total_docs,
        "trang tiếp theo": ckpt.next_page,
        "file": paths.docs,
    })
    return 0


def cmd_reconcile(args) -> int:
    paths = Paths(Path(args.data_dir))
    _setup_logging(args.verbose, paths.log)
    try:
        ckpt = Checkpoint.load(paths.checkpoint, LIST_QUERY, args.crawl_page_size)
    except CheckpointMismatch as e:
        log.error("%s", e)
        return 2
    client = VbplClient(delay=args.delay)
    with JsonlStore(paths.docs) as store:
        try:
            stats = reconcile(client, store, ckpt, page_size=args.page_size,
                              max_pages=args.max_pages, workers=args.workers)
        except CrawlAborted as e:
            log.error("DỪNG: %s", e)
            return 2
    _print_kv("Kết quả reconcile", {
        "trang danh sách đã đọc": stats.list_pages,
        "văn bản trong danh sách": stats.listed,
        "thiếu trong docs.jsonl": stats.missing,
        "đã crawl bù": stats.fetched,
        "crawl bù lỗi": stats.failed,
        "liệt kê đủ toàn bộ": stats.complete,
        "có trong file nhưng không còn trong danh sách": stats.extra_in_store if stats.complete else "(chỉ tính khi liệt kê đủ)",
    })
    return 0


def cmd_status(args) -> int:
    paths = Paths(Path(args.data_dir))
    if not paths.checkpoint.exists():
        print(f"Chưa có checkpoint tại {paths.checkpoint}. Chạy `python -m legal_rag crawl` trước.")
        return 1
    data = Checkpoint.read(paths.checkpoint)
    with JsonlStore(paths.docs) as store:
        n_docs = len(store)
    _print_kv("Trạng thái crawl", {
        "văn bản trong docs.jsonl": n_docs,
        "total theo API": data.get("total"),
        "trang tiếp theo / trang cuối": f"{data.get('next_page')} / {data.get('last_page')}",
        "đã hết danh sách": data.get("done"),
        "trang lỗi": len(data.get("failed_pages", {})),
        "văn bản lỗi": len(data.get("failed_docs", {})),
        "lý do lỗi văn bản": dict(Counter(data.get("failed_docs", {}).values())) or "-",
        "cập nhật lúc (UTC)": data.get("updated_at"),
    })
    return 0


def cmd_preprocess(args) -> int:
    paths = Paths(Path(args.data_dir))
    _setup_logging(args.verbose)
    if not paths.docs.exists():
        log.error("Không thấy %s. Chạy crawl trước.", paths.docs)
        return 1
    report = run_preprocess(paths.docs, paths.processed_dir)
    _print_kv("Kết quả preprocess", {
        "dòng trong docs.jsonl": report.raw_lines,
        "dòng hỏng": report.bad_lines,
        "bản trùng đã bỏ": report.duplicates_removed,
        "văn bản giữ lại": report.kept,
        "văn bản bị loại": dict(report.rejected) or "-",
        "node theo cấp": dict(report.levels),
        "dòng nối tiếp cùng id": f"{report.merged_rows} (ở {report.docs_with_merged_rows} văn bản)",
        "class prov-* lạ": dict(report.unknown_classes) or "-",
        "output": paths.processed_dir,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m legal_rag", description="Crawl + preprocess VBPL")
    parser.add_argument("--data-dir", default="data", help="thư mục dữ liệu (mặc định: ./data)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="gọi thử 1 trang + 1 văn bản, không ghi file")
    p.add_argument("--n", type=int, default=3, help="số mục danh sách lấy thử")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("crawl", help="Phase 1: crawl (mặc định chỉ demo 2 trang)")
    p.add_argument("--max-pages", type=int, default=2, help="số trang tối đa cho lần chạy này (mặc định 2)")
    p.add_argument("--max-docs", type=int, default=None, help="số văn bản tối đa cho lần chạy này")
    p.add_argument("--all", action="store_true", help="bỏ giới hạn trang, crawl tới hết danh sách")
    p.add_argument("--page-size", type=int, default=10)
    p.add_argument("--workers", type=int, default=1, help="số luồng gọi detail song song")
    p.add_argument("--delay", type=float, default=0.3, help="giây nghỉ trước mỗi request (mỗi luồng)")
    p.add_argument("--skip-invalid-status", action="store_true",
                   help="không gọi detail cho văn bản có hiệu lực sẽ bị Phase 2 loại")
    p.add_argument("--retry-failed", action="store_true", help="chỉ thử lại trang/văn bản lỗi trong checkpoint")
    p.add_argument("--reset", action="store_true", help="bỏ checkpoint cũ (docs.jsonl vẫn giữ, không ghi trùng)")
    p.set_defaults(func=cmd_crawl)

    p = sub.add_parser("reconcile", help="liệt kê lại danh sách và crawl bù văn bản bị sót")
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--crawl-page-size", type=int, default=10, help="page_size đã dùng khi crawl (để mở checkpoint)")
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--delay", type=float, default=0.3)
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("status", help="xem tiến độ crawl")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("preprocess", help="Phase 2: khử trùng, lọc, dựng cây mục lục")
    p.set_defaults(func=cmd_preprocess)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except VbplError as e:
        log.error("Lỗi API: %s", e)
        print(f"Lỗi API: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
