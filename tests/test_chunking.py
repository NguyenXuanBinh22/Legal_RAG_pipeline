import json
import sqlite3
import unicodedata

import pytest

from legal_rag.chunking import article_number, build_chunks, run_chunk
from legal_rag.structure import build_provision_tree
from tests.conftest import SAMPLE_HTML, make_detail, make_html


def chunks_of(html, title="Luật mẫu"):
    tree, _ = build_provision_tree(html)
    return build_chunks(title, tree, html)


def test_sample_document_leaf_chunks():
    chunks = {c.chunk_id: c for c in chunks_of(SAMPLE_HTML)}
    assert list(chunks) == ["a1", "k1", "pa", "pb"]  # Chương/Điều 17/Khoản 2 không phải lá

    assert chunks["a1"].embed_text == "Luật mẫu - Điều 1. Phạm vi điều chỉnh: Luật này quy định về việc thành lập doanh nghiệp."
    assert (chunks["a1"].article, chunks["a1"].clause) == ("Điều 1", None)

    assert chunks["k1"].embed_text == ("Luật mẫu - Điều 17. Quyền thành lập, góp vốn, mua cổ phần - Khoản 1: "
                                       "1. Tổ chức, cá nhân có quyền thành lập doanh nghiệp.")
    assert chunks["pa"].embed_text == ("Luật mẫu - Điều 17. Quyền thành lập, góp vốn, mua cổ phần - Khoản 2 - Điểm a: "
                                       "a) Cơ quan nhà nước sử dụng tài sản nhà nước;")
    assert (chunks["pa"].article, chunks["pa"].clause) == ("Điều 17", "Khoản 2")


def test_rows_with_same_id_are_one_chunk():
    # Hình dạng thật: dòng nối tiếp data-merged và prov-content mang id của Điều
    html = (
        '<p class="prov-article" id="a1">Điều 292. Biện pháp bảo đảm</p>'
        '<p class="prov-article" id="a1" data-merged="true">thực hiện nghĩa vụ</p>'
        '<p class="prov-content" id="a1">Các biện pháp bảo đảm gồm cầm cố, thế chấp.</p>'
        '<p class="prov-article" id="a2">Điều 293. Phạm vi</p>'
        '<p class="prov-clause" id="k1">1. Nghĩa vụ có thể được bảo đảm</p>'
        '<p class="prov-content" id="k1">một phần hoặc toàn bộ.</p>'
    )
    chunks = chunks_of(html)
    assert [c.chunk_id for c in chunks] == ["a1", "k1"]
    assert chunks[0].embed_text == ("Luật mẫu - Điều 292. Biện pháp bảo đảm thực hiện nghĩa vụ: "
                                    "Các biện pháp bảo đảm gồm cầm cố, thế chấp.")
    assert chunks[1].embed_text.endswith("- Khoản 1: 1. Nghĩa vụ có thể được bảo đảm một phần hoặc toàn bộ.")


def test_trailing_rows_without_id_belong_to_previous_leaf():
    html = make_html([
        ("prov-article", "a1", "Điều 5. Mức phí"),
        ("prov-clause", "k1", "1. Phí tính theo công thức P = S0 x 0,15%"),
        ("prov-content", None, "Trong đó:"),
        ("prov-content", None, "- P: là số phí phải nộp."),
        ("prov-chapter", "c2", "Chương II"),
        ("prov-content", None, "đoạn sau Chương, không thuộc Khoản 1"),
    ])
    k1 = next(c for c in chunks_of(html) if c.chunk_id == "k1")
    assert k1.embed_text.endswith("Phí tính theo công thức P = S0 x 0,15% Trong đó: - P: là số phí phải nộp.")
    assert "không thuộc" not in k1.embed_text


def test_article_without_content_is_title_only():
    html = make_html([("prov-article", "a1", "Điều 3. (được bãi bỏ)")])
    [chunk] = chunks_of(html)
    assert chunk.embed_text == "Luật mẫu - Điều 3. (được bãi bỏ)"
    assert not chunk.has_content


def test_text_is_nfc():
    html = make_html([("prov-article", "a1", unicodedata.normalize("NFD", "Điều 1. Quyền lợi"))])
    [chunk] = chunks_of(html)
    assert chunk.embed_text == unicodedata.normalize("NFC", chunk.embed_text)
    assert chunk.article == "Điều 1"


@pytest.mark.parametrize("title, expected", [
    ("Điều 5. Phạm vi", "Điều 5"),
    ("ĐIỀU 12 PHẠM VI", "Điều 12"),
    ("Điều 5a. Bổ sung", "Điều 5a"),
    ("Điều 1Phạm vi", "Điều 1"),
    ("Chương I", None),
])
def test_article_number(title, expected):
    assert article_number(title) == expected


def write_processed(path, items):
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def test_run_chunk_writes_search_service_schema(tmp_path):
    processed = tmp_path / "processed_data.json"
    good = {**make_detail("A"), "metadata": build_provision_tree(SAMPLE_HTML)[0]}
    write_processed(processed, [
        good,
        {**make_detail("B"), "metadata": []},
        {**make_detail("C", html=None), "metadata": good["metadata"]},
        good,  # trùng doc_id
        {**make_detail("D"), "metadata": build_provision_tree(SAMPLE_HTML)[0]},
    ])
    report = run_chunk(processed, tmp_path / "index")

    db = sqlite3.connect(tmp_path / "index" / "chunks.db")
    rows = db.execute("SELECT faiss_id, chunk_id, doc_id, doc_num, title, article, clause FROM chunks").fetchall()
    assert [r[0] for r in rows] == list(range(8))
    assert [(r[1], r[2]) for r in rows[:4]] == [("a1", "A"), ("k1", "A"), ("pa", "A"), ("pb", "A")]
    assert rows[2][3:] == ("A/2020/QH14", "Luật mẫu A", "Điều 17", "Khoản 2")
    assert db.execute("SELECT doc_id, doc_num_norm, title_norm FROM docs").fetchall() == [
        ("A", "a/2020/qh14", "luật mẫu a"), ("D", "d/2020/qh14", "luật mẫu d")]
    indexes = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"idx_chunks_doc_id", "idx_docs_doc_num_norm"} <= indexes
    db.close()

    assert report.chunks == 8
    assert report.docs_indexed == 2
    assert report.duplicate_chunk_ids == 4  # D dùng lại id phần tử của A
    assert dict(report.rejected) == {"no_metadata": 1, "no_content": 1, "duplicate_doc": 1}
    errors = json.loads((tmp_path / "index" / "chunk_errors.json").read_text(encoding="utf-8"))
    assert [e["doc_id"] for e in errors] == ["B", "C", "A"]


def test_failed_run_keeps_previous_db(tmp_path):
    processed = tmp_path / "processed_data.json"
    write_processed(processed, [{**make_detail("A"), "metadata": build_provision_tree(SAMPLE_HTML)[0]}])
    run_chunk(processed, tmp_path / "index")
    before = (tmp_path / "index" / "chunks.db").read_bytes()

    processed.write_text('[{"id": "A", "title": ', encoding="utf-8")  # JSON cụt
    with pytest.raises(Exception):
        run_chunk(processed, tmp_path / "index")
    assert (tmp_path / "index" / "chunks.db").read_bytes() == before
    assert not (tmp_path / "index" / "chunks.db.tmp").exists()
