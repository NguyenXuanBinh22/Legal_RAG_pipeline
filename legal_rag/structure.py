"""Dựng cây mục lục Phần/Chương/Mục/Điều/Khoản/Điểm từ HTML của API detail.

HTML của VBPL đánh dấu từng đơn vị bằng class `prov-*` và `id`, xếp PHẲNG theo thứ tự
đọc (Khoản, Điểm là anh em của Điều chứ không lồng bên trong). Cây được dựng bằng một
stack theo thứ bậc cấp. Schema node giữ tương thích với `metadata` mà
chunk_embedding.py (Phase 3) đang đọc: id, title, level, ptype, orderIndex, children, isLeaf.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

# class CSS → (level, thứ bậc). Thứ bậc nhỏ hơn = cấp cao hơn.
PROV_LEVELS = {
    "prov-part": ("Part", 0),
    "prov-chapter": ("Chapter", 1),
    "prov-section": ("Section", 2),
    "prov-subsection": ("SubSection", 3),
    "prov-article": ("Article", 4),
    "prov-clause": ("Clause", 5),
    "prov-item": ("Point", 6),
}
# Chỉ chứa nội dung (đoạn dẫn, Điều không có Khoản), không tạo node
CONTENT_CLASSES = {"prov-content"}

_TITLE_RULES = {
    "Part": re.compile(r"^(Phần\s+(?:thứ\s+)?\S+)", re.IGNORECASE),
    "Chapter": re.compile(r"^(Chương\s+[IVXLCDM\d]+[A-Z]?)", re.IGNORECASE),
    "Section": re.compile(r"^(Mục\s+\d+[a-z]?)", re.IGNORECASE),
    "SubSection": re.compile(r"^(Tiểu\s+mục\s+\d+)", re.IGNORECASE),
    "Article": re.compile(r"^(Điều\s+\d+[a-zđ]?)", re.IGNORECASE),
}
_CLAUSE_NUM = re.compile(r"^(\d+[a-zđ]?)\s*[.)]")
_POINT_LETTER = re.compile(r"^([a-zđ]{1,2}\d?)\s*\)")
_FALLBACK_TITLE_LEN = 60


@dataclass
class TreeStats:
    levels: Counter = field(default_factory=Counter)
    skipped_no_id: int = 0
    # VBPL tách 1 đơn vị thành nhiều <p> cùng id (dòng sau có data-merged="true"),
    # vd. "Mục 1" + "NĂNG LỰC PHÁP LUẬT DÂN SỰ…". Các dòng nối tiếp không tạo node mới.
    merged_rows: int = 0
    unknown_classes: Counter = field(default_factory=Counter)


def normalize_text(text: str) -> str:
    """NFC + gộp khoảng trắng: HTML có thể chứa tiếng Việt dạng tổ hợp (NFD)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def node_title(level: str, text: str) -> str:
    # Tiêu đề Khoản/Điểm đi vào metadata `clause` và header embed_text ở Phase 3,
    # nên đoạn không đánh số dùng nhãn cố định thay vì cắt nội dung
    if level == "Clause":
        match = _CLAUSE_NUM.match(text)
        return f"Khoản {match.group(1)}" if match else "Khoản không đánh số"
    if level == "Point":
        match = _POINT_LETTER.match(text)
        return f"Điểm {match.group(1)}" if match else "Điểm không đánh số"
    rule = _TITLE_RULES.get(level)
    match = rule.match(text) if rule else None
    return match.group(1) if match else text[:_FALLBACK_TITLE_LEN]


def _has_prov_class(tag) -> bool:
    classes = tag.get("class")
    return bool(classes) and any(c.startswith("prov-") for c in classes)


def build_provision_tree(html: str) -> tuple[list[dict], TreeStats]:
    stats = TreeStats()
    if not html:
        return [], stats

    soup = BeautifulSoup(html, "html.parser")
    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []
    seen_ids: set[str] = set()

    for el in soup.find_all(_has_prov_class):
        classes = [c for c in el.get("class", []) if c.startswith("prov-")]
        known = [c for c in classes if c in PROV_LEVELS]
        if not known:
            for c in classes:
                if c not in CONTENT_CLASSES:
                    stats.unknown_classes[c] += 1
            continue

        el_id = el.get("id")
        if not el_id:
            stats.skipped_no_id += 1
            continue
        if el_id in seen_ids:
            stats.merged_rows += 1
            continue
        seen_ids.add(el_id)

        level, rank = PROV_LEVELS[known[0]]
        while stack and stack[-1][0] >= rank:
            stack.pop()
        siblings = stack[-1][1]["children"] if stack else roots
        node = {
            "id": el_id,
            "title": node_title(level, normalize_text(el.get_text(" ", strip=True))),
            "level": level,
            "ptype": level,
            "orderIndex": len(siblings) + 1,
            "children": [],
            "isLeaf": True,
        }
        siblings.append(node)
        if stack:
            stack[-1][1]["isLeaf"] = False
        stack.append((rank, node))
        stats.levels[level] += 1

    return roots, stats
