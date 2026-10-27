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

# Phase 4: API embedding kiểu OpenAI (LM Studio…). Biến môi trường/.env cùng tên với repo cũ
# (EMBEDDING_BASE_URL, EMBEDDING_MODEL_NAME, EMBEDDING_API_KEY) được ưu tiên hơn giá trị này.
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:1234/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"


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

    @property
    def processed_data(self) -> Path:
        return self.processed_dir / "processed_data.json"

    # Phase 3-4: đúng tên file mà SearchService (services/Search.py) đọc trong data/
    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def chunks_db(self) -> Path:
        return self.index_dir / "chunks.db"

    @property
    def faiss_index(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def faiss_fp16_index(self) -> Path:
        return self.index_dir / "faiss_fp16.index"

    @property
    def embed_skipped(self) -> Path:
        return self.index_dir / "embed_skipped.json"
