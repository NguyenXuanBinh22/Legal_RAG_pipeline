"""Phase 2: khử trùng, lọc hiệu lực, dựng cây mục lục → processed_data.json cho Phase 3."""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import KEEP_STATUSES
from .storage import atomic_write_json, extract_doc_id
from .structure import build_provision_tree

log = logging.getLogger(__name__)


@dataclass
class PreprocessReport:
    raw_lines: int = 0
    bad_lines: int = 0
    unique_docs: int = 0
    duplicates_removed: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)
    levels: Counter = field(default_factory=Counter)
    unknown_classes: Counter = field(default_factory=Counter)
    nodes_without_id: int = 0
    merged_rows: int = 0
    docs_with_merged_rows: int = 0

    def to_dict(self) -> dict:
        # dataclasses.asdict dựng lại Counter từ các cặp (k, v) → đếm nhầm tuple làm key
        return {k: dict(v) if isinstance(v, Counter) else v for k, v in vars(self).items()}


class _JsonArrayWriter:
    """Ghi JSON array từng phần tử (không giữ cả mảng trong RAM), hoàn tất bằng os.replace."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.count = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.tmp, "w", encoding="utf-8", newline="\n")
        self._f.write("[")
        return self

    def write(self, obj) -> None:
        self._f.write(",\n" if self.count else "\n")
        self._f.write(json.dumps(obj, ensure_ascii=False))
        self.count += 1

    def __exit__(self, exc_type, *_):
        self._f.write("\n]\n" if self.count else "]\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        if exc_type is None:
            os.replace(self.tmp, self.path)
        else:
            self.tmp.unlink(missing_ok=True)


def index_last_occurrence(path: Path) -> tuple[dict[str, int], int, int]:
    """Offset dòng CUỐI của mỗi doc_id (bản crawl mới nhất thắng). Trả về (offsets, số dòng, số dòng hỏng)."""
    offsets: dict[str, int] = {}
    lines = bad = 0
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            lines += 1
            doc_id = extract_doc_id(line)
            if doc_id is None:
                bad += 1
                continue
            offsets[doc_id] = offset
    return offsets, lines, bad


def reject_reason(detail, keep: set[str]) -> str | None:
    if not isinstance(detail, dict) or not detail:
        return "detail_missing"
    eff = detail.get("effStatus")
    if not isinstance(eff, dict) or not eff.get("name"):
        return "missing_effStatus"
    name = unicodedata.normalize("NFC", eff["name"])
    if name not in keep:
        return f"unsupported_effStatus: {name}"
    content = detail.get("documentContent")
    if not isinstance(content, dict) or not content.get("content"):
        return "no_content"
    return None


def _error_item(doc_id: str, detail, reason: str) -> dict:
    detail = detail if isinstance(detail, dict) else {}
    eff = detail.get("effStatus")
    return {
        "doc_id": doc_id,
        "docNum": detail.get("docNum"),
        "title": detail.get("title"),
        "effStatus": eff.get("name") if isinstance(eff, dict) else None,
        "effFrom": detail.get("effFrom"),
        "reason": reason,
    }


def run_preprocess(raw_path: Path, out_dir: Path, keep_statuses=KEEP_STATUSES) -> PreprocessReport:
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    keep = {unicodedata.normalize("NFC", s) for s in keep_statuses}
    report = PreprocessReport()

    offsets, report.raw_lines, report.bad_lines = index_last_occurrence(raw_path)
    report.unique_docs = len(offsets)
    report.duplicates_removed = report.raw_lines - report.bad_lines - report.unique_docs

    errors: list[dict] = []
    with open(raw_path, "rb") as raw, _JsonArrayWriter(out_dir / "processed_data.json") as out:
        for doc_id, offset in sorted(offsets.items(), key=lambda kv: kv[1]):
            raw.seek(offset)
            detail = json.loads(raw.readline()).get("detail")

            reason = reject_reason(detail, keep)
            tree = []
            if reason is None:
                tree, tree_stats = build_provision_tree(detail["documentContent"]["content"])
                report.levels.update(tree_stats.levels)
                report.unknown_classes.update(tree_stats.unknown_classes)
                report.nodes_without_id += tree_stats.skipped_no_id
                report.merged_rows += tree_stats.merged_rows
                report.docs_with_merged_rows += int(tree_stats.merged_rows > 0)
                if not tree:
                    reason = "no_structure"

            if reason is not None:
                report.rejected[reason.split(":")[0]] += 1
                errors.append(_error_item(doc_id, detail, reason))
                continue

            # Giữ nguyên toàn bộ detail + thêm "metadata" = cây mục lục, đúng định dạng Phase 3 đọc
            out.write({**detail, "metadata": tree})
            report.kept += 1

    atomic_write_json(out_dir / "error_items.json", errors)
    atomic_write_json(out_dir / "report.json", report.to_dict())
    return report
