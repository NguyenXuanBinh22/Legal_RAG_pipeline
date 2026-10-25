"""Lưu trữ chịu được crash: JSONL append-only cho dữ liệu, checkpoint ghi nguyên tử."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Mỗi bản ghi được ghi với doc_id là key đầu tiên → đọc lại id không cần parse cả dòng
# (một dòng chứa toàn bộ HTML văn bản, có thể dài hàng trăm KB).
_DOC_ID_PREFIX = re.compile(rb'^\{"doc_id": ("(?:[^"\\]|\\.)*")')


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj: Any) -> None:
    """Ghi ra file .tmp rồi os.replace: người đọc không bao giờ thấy file ghi dở."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def extract_doc_id(line: bytes) -> str | None:
    match = _DOC_ID_PREFIX.match(line)
    if match:
        return json.loads(match.group(1))
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    value = obj.get("doc_id") if isinstance(obj, dict) else None
    return str(value) if value is not None else None


def iter_doc_ids(path: Path) -> Iterator[str]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                doc_id = extract_doc_id(line)
                if doc_id is not None:
                    yield doc_id


def repair_trailing_partial_line(path: Path) -> int:
    """Cắt dòng cuối bị ghi dở do crash. Trả về số byte đã cắt (0 nếu file lành)."""
    path = Path(path)
    if not path.exists():
        return 0
    size = path.stat().st_size
    if size == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return 0
        cut, pos, block = 0, size, 1 << 16
        while pos > 0:
            start = max(0, pos - block)
            f.seek(start)
            idx = f.read(pos - start).rfind(b"\n")
            if idx != -1:
                cut = start + idx + 1
                break
            pos = start
        f.truncate(cut)
    return size - cut


class JsonlStore:
    """docs.jsonl: 1 dòng = 1 văn bản. Giữ tập doc_id đã ghi để khử trùng khi resume."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired_bytes = repair_trailing_partial_line(self.path)
        self.seen: set[str] = set(iter_doc_ids(self.path))
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.seen

    def __len__(self) -> int:
        return len(self.seen)

    def append(self, record: dict) -> None:
        doc_id = str(record["doc_id"])
        ordered = {"doc_id": doc_id, **{k: v for k, v in record.items() if k != "doc_id"}}
        self._f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
        self.seen.add(doc_id)

    def flush(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        if not self._f.closed:
            self.flush()
            self._f.close()

    def __enter__(self) -> "JsonlStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CheckpointMismatch(Exception):
    pass


@dataclass
class Checkpoint:
    path: Path
    query: dict
    page_size: int
    next_page: int = 1
    total: int | None = None
    last_page: int | None = None
    done: bool = False
    failed_pages: dict[str, str] = field(default_factory=dict)
    failed_docs: dict[str, str] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def load(cls, path: Path, query: dict, page_size: int, reset: bool = False) -> "Checkpoint":
        path = Path(path)
        if reset or not path.exists():
            return cls(path=path, query=dict(query), page_size=page_size)
        data = cls.read(path)
        if data.get("query") != query or data.get("page_size") != page_size:
            raise CheckpointMismatch(
                f"Checkpoint {path} được tạo với page_size={data.get('page_size')}, "
                f"query={data.get('query')}. Số trang sẽ bị lệch nếu đổi tham số. "
                "Giữ nguyên tham số cũ, hoặc dùng --reset (docs.jsonl vẫn được giữ và khử trùng)."
            )
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k != "path" and k in data}
        return cls(path=path, **fields)

    @staticmethod
    def read(path: Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def update_total(self, total: int | None) -> None:
        if total is not None:
            self.total = total
            self.last_page = math.ceil(total / self.page_size)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("path")
        return data

    def save(self) -> None:
        self.updated_at = now_iso()
        atomic_write_json(self.path, self.to_dict())
