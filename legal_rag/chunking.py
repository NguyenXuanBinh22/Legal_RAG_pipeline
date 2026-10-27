"""Phase 3: processed_data.json → chunks.db, mỗi chunk là một đơn vị LÁ của cây mục lục.

Định dạng embed_text giữ như chunk_embedding.py cũ:
    "<tên văn bản> - <tiêu đề Điều> - <Mục/Khoản/Điểm…>: <nội dung>"
(bỏ Phần/Chương khỏi chuỗi cấp bậc), metadata gồm title, doc_num, doc_id, article, clause.

Khác code cũ: mọi dòng HTML của một đơn vị được gộp theo thứ tự đọc. VBPL tách một đơn vị
thành nhiều phần tử CÙNG id (dòng nối tiếp data-merged="true", hoặc prov-content mang id của
Điều/Khoản), và đặt đoạn không id (vd. "Trong đó: - P: là…") ngay sau đơn vị chứa nó.
Code cũ giữ phần tử cuối theo id → chunk trùng chunk_id, tiêu đề Điều bị thay bằng dòng sau,
và bỏ mất đoạn không id sau Khoản/Điểm.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import ijson
from bs4 import BeautifulSoup

from .chunk_db import ChunkDbWriter
from .storage import atomic_write_json
from .structure import PROV_LEVELS, normalize_text

log = logging.getLogger(__name__)

LEVEL_CLASS = {level: cls for cls, (level, _) in PROV_LEVELS.items()}
# Gặp đơn vị cấu trúc không có trong cây → đoạn không id phía sau không thuộc đơn vị trước nữa
STRUCTURAL_CLASSES = {"prov-part", "prov-chapter", "prov-section", "prov-subsection", "prov-article"}
BODY_CLASSES = {"prov-content", "prov-clause", "prov-item"}
# Điều đã có tiêu đề đầy đủ riêng trong embed_text
HIERARCHY_SKIP = {"Part", "Chapter", "Article"}

# Hậu tố chữ chỉ nhận khi không dính liền chữ khác: "Điều 5a." → 5a, "Điều 1Phạm vi" → 1
_ARTICLE_NUM = re.compile(r"Điều\s+(\d+)([a-zđ](?![^\W\d_]))?", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    embed_text: str
    article: str | None
    clause: str | None
    has_content: bool = True


@dataclass
class _Unit:
    heading: list[str] = field(default_factory=list)  # dòng mang đúng class của cấp đó
    body: list[str] = field(default_factory=list)     # dòng cùng id khác class + đoạn không id theo sau


def article_number(title: str) -> str | None:
    match = _ARTICLE_NUM.match(title)
    if not match:
        return None
    return f"Điều {match.group(1)}{(match.group(2) or '').lower()}"


def _flatten(nodes: list[dict], chain: tuple = ()) -> dict[str, tuple[dict, tuple]]:
    """id → (node, chuỗi node từ gốc tới chính nó)"""
    out = {}
    for node in nodes:
        full = (*chain, node)
        out[node["id"]] = (node, full)
        out.update(_flatten(node.get("children") or [], full))
    return out


def _is_leaf(node: dict) -> bool:
    return bool(node.get("isLeaf")) or not node.get("children")


def _collect_units(html: str, nodes: dict) -> dict[str, _Unit]:
    """Gom text từng đơn vị theo thứ tự đọc. HTML VBPL xếp phẳng (Khoản là anh em của Điều)."""
    soup = BeautifulSoup(html, "html.parser")
    units: dict[str, _Unit] = {}
    current: _Unit | None = None
    for el in soup.find_all(lambda t: t.get("id") in nodes or any(c.startswith("prov-") for c in t.get("class") or [])):
        classes = set(el.get("class") or [])
        el_id = el.get("id")
        text = normalize_text(el.get_text(" ", strip=True))
        if el_id in nodes:
            current = units.setdefault(el_id, _Unit())
            target = current.heading if LEVEL_CLASS.get(nodes[el_id][0]["level"]) in classes else current.body
        elif classes & STRUCTURAL_CLASSES:
            current = None
            continue
        elif current is not None and classes & BODY_CLASSES:
            target = current.body
        else:
            continue
        if text:
            target.append(text)
    return units


def _strip_title(text: str, title: str) -> str:
    if title and text.startswith(title):
        return text[len(title):].strip().lstrip(".:").strip()
    return text


def build_chunks(doc_title: str, tree: list[dict], html: str) -> list[Chunk]:
    nodes = _flatten(tree)
    units = _collect_units(html, nodes)
    chunks = []
    for node_id, unit in units.items():  # dict giữ thứ tự xuất hiện trong HTML
        node, chain = nodes[node_id]
        if not _is_leaf(node):
            continue

        article = next((n for n in reversed(chain) if n["level"] == "Article"), None)
        art_title = ""
        if article is not None:
            art_unit = units.get(article["id"])
            art_title = " ".join(art_unit.heading) if art_unit and art_unit.heading else normalize_text(article["title"])

        title = normalize_text(node["title"])
        if node["level"] == "Article":
            content = " ".join(unit.body)
        else:
            content = _strip_title(" ".join(unit.heading + unit.body), title)

        hierarchy = " - ".join(normalize_text(n["title"]) for n in chain if n["level"] not in HIERARCHY_SKIP)
        embed_text = " - ".join(p for p in (doc_title, art_title, hierarchy) if p)
        if content:
            embed_text += f": {content}"
        if not embed_text:
            continue

        clause = next((normalize_text(n["title"]) for n in chain if n["level"] == "Clause"), None)
        chunks.append(Chunk(node_id, embed_text, article_number(art_title) if art_title else None, clause,
                            has_content=bool(content)))
    return chunks


@dataclass
class ChunkReport:
    docs_read: int = 0
    docs_indexed: int = 0
    chunks: int = 0
    title_only_chunks: int = 0  # đơn vị lá không có nội dung, embed_text chỉ có tiêu đề
    duplicate_chunk_ids: int = 0
    longest_chunk_chars: int = 0
    rejected: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {k: dict(v) if isinstance(v, Counter) else v for k, v in vars(self).items()}


def run_chunk(processed_path: Path, out_dir: Path) -> ChunkReport:
    """Đọc processed_data.json dạng stream (không nạp cả mảng vào RAM), ghi out_dir/chunks.db."""
    out_dir = Path(out_dir)
    report = ChunkReport()
    errors: list[dict] = []
    seen_docs: set[str] = set()
    seen_chunk_ids: set[str] = set()

    with open(processed_path, "rb") as f, ChunkDbWriter(out_dir / "chunks.db") as db:
        for item in ijson.items(f, "item", use_float=True):
            report.docs_read += 1
            doc_id = str(item.get("id") or "")
            # NFC: SearchService so khớp số hiệu/tên văn bản với chuỗi người dùng gõ
            doc_num = normalize_text(item.get("docNum") or "") or None
            title = normalize_text(item.get("title") or "") or None
            html = (item.get("documentContent") or {}).get("content")
            tree = item.get("metadata")

            reason = None
            chunks: list[Chunk] = []
            if not doc_id:
                reason = "missing_id"
            elif doc_id in seen_docs:
                reason = "duplicate_doc"
            elif not html:
                reason = "no_content"
            elif not tree:
                reason = "no_metadata"
            else:
                try:
                    chunks = build_chunks(title or "", tree, html)
                except (KeyError, TypeError) as e:  # cây mục lục sai schema
                    reason = f"bad_metadata: {type(e).__name__}: {e}"[:200]
                else:
                    if not chunks:
                        reason = "no_chunks"

            if reason is not None:
                report.rejected[reason.split(":")[0]] += 1
                errors.append({"doc_id": doc_id, "docNum": doc_num, "title": title, "reason": reason})
                continue

            seen_docs.add(doc_id)
            db.add_doc(doc_id, doc_num, title)
            for chunk in chunks:
                if chunk.chunk_id in seen_chunk_ids:
                    report.duplicate_chunk_ids += 1
                seen_chunk_ids.add(chunk.chunk_id)
                if not chunk.has_content:
                    report.title_only_chunks += 1
                report.longest_chunk_chars = max(report.longest_chunk_chars, len(chunk.embed_text))
                db.add_chunk(chunk.chunk_id, doc_id, doc_num, title, chunk.article, chunk.clause, chunk.embed_text)
            report.docs_indexed += 1

            if report.docs_read % 1000 == 0:
                log.info("[%s văn bản] %s chunk", report.docs_read, db.count)

        report.chunks = db.count

    atomic_write_json(out_dir / "chunk_errors.json", errors)
    atomic_write_json(out_dir / "chunk_report.json", report.to_dict())
    return report
