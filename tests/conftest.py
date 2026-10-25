"""Fixture dùng chung: HTML mẫu, client giả lập API (không gọi mạng)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.config import LIST_QUERY
from legal_rag.storage import Checkpoint, JsonlStore
from legal_rag.vbpl_client import ListPage, VbplError

# Cấu trúc thật của HTML VBPL: các đơn vị là anh em phẳng, theo thứ tự đọc
SAMPLE_ROWS = [
    ("prov-chapter", "c1", "Chương I QUY ĐỊNH CHUNG"),
    ("prov-article", "a1", "Điều 1. Phạm vi điều chỉnh"),
    ("prov-content", None, "Luật này quy định về việc thành lập doanh nghiệp."),
    ("prov-chapter", "c2", "Chương II THÀNH LẬP DOANH NGHIỆP"),
    ("prov-article", "a17", "Điều 17. Quyền thành lập, góp vốn, mua cổ phần"),
    ("prov-clause", "k1", "1. Tổ chức, cá nhân có quyền thành lập doanh nghiệp."),
    ("prov-clause", "k2", "2. Tổ chức, cá nhân sau đây không có quyền thành lập doanh nghiệp:"),
    ("prov-item", "pa", "a) Cơ quan nhà nước sử dụng tài sản nhà nước;"),
    ("prov-item", "pb", "b) Cán bộ, công chức, viên chức;"),
]


def make_html(rows) -> str:
    parts = []
    for cls, el_id, text in rows:
        id_attr = f' id="{el_id}"' if el_id else ""
        parts.append(f'<p class="{cls}"{id_attr}>{text}</p>')
    return "<html><body>" + "".join(parts) + "</body></html>"


SAMPLE_HTML = make_html(SAMPLE_ROWS)


def make_detail(doc_id: str, status: str | None = "Còn hiệu lực", html: str | None = SAMPLE_HTML) -> dict:
    return {
        "id": doc_id,
        "docNum": f"{doc_id}/2020/QH14",
        "title": f"Luật mẫu {doc_id}",
        "effStatus": {"code": "X", "name": status} if status else None,
        "effFrom": "2021-01-01T00:00:00",
        "documentContent": {"id": "x", "title": "", "content": html} if html is not None else None,
    }


class FakeClient:
    """Giả lập VbplClient trên một danh mục cố định. Lỗi lập trình được qua page_errors/doc_errors."""

    def __init__(self, n_docs: int = 25, status_for=lambda i: "Còn hiệu lực"):
        self.items = [{"id": str(1000 + i), "effStatus": {"name": status_for(i)}} for i in range(n_docs)]
        self.details = {it["id"]: make_detail(it["id"], it["effStatus"]["name"]) for it in self.items}
        self.page_errors: dict[int, str] = {}
        self.doc_errors: dict[str, str] = {}
        self.list_calls: list[tuple[int, int]] = []
        self.detail_calls: list[str] = []

    def list_page(self, page_number: int, page_size: int) -> ListPage:
        self.list_calls.append((page_number, page_size))
        if page_number in self.page_errors:
            raise VbplError(self.page_errors[page_number])
        start = (page_number - 1) * page_size
        return ListPage(page_number, page_size, len(self.items), self.items[start:start + page_size])

    def get_detail(self, doc_id: str) -> dict:
        self.detail_calls.append(doc_id)
        if doc_id in self.doc_errors:
            raise VbplError(self.doc_errors[doc_id])
        return self.details[doc_id]


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return tmp_path / "raw"


def open_store(raw_dir: Path) -> JsonlStore:
    return JsonlStore(raw_dir / "docs.jsonl")


def open_ckpt(raw_dir: Path, page_size: int = 10) -> Checkpoint:
    return Checkpoint.load(raw_dir / "checkpoint.json", LIST_QUERY, page_size)


def read_doc_ids(path: Path) -> list[str]:
    return [json.loads(line)["doc_id"] for line in path.read_text(encoding="utf-8").splitlines() if line]
