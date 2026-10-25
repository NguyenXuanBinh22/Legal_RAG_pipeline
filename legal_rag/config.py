"""Hằng số và đường dẫn dùng chung."""

from dataclasses import dataclass
from pathlib import Path

GATEWAY_BASE = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc"
USER_AGENT = "LegalRAG-crawler/0.1 (+research; python-requests)"

# Giữ đúng truy vấn của code cũ: sortByViewCount=True trả về tập ~37.673 văn bản và
# bỏ qua sortBy. Thứ tự theo lượt xem KHÔNG ổn định giữa các lần gọi, nên crawl dựa vào
# khử trùng theo doc_id + lệnh `reconcile` để bù văn bản bị trượt trang.
LIST_QUERY = {"sortDirection": "desc", "sortBy": "viewCount", "sortByViewCount": True}

KEEP_STATUSES = ("Còn hiệu lực", "Hết hiệu lực một phần", "Chưa có hiệu lực")


@dataclass(frozen=True)
class Paths:
    data_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def docs(self) -> Path:
        return self.raw_dir / "docs.jsonl"

    @property
    def checkpoint(self) -> Path:
        return self.raw_dir / "checkpoint.json"

    @property
    def log(self) -> Path:
        return self.raw_dir / "crawl.log"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
