import json

from legal_rag.preprocess import run_preprocess
from tests.conftest import make_detail, make_html


def write_raw(path, records, garbage_lines=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for doc_id, detail in records:
            f.write(json.dumps({"doc_id": doc_id, "page": 1, "list_item": None, "detail": detail},
                               ensure_ascii=False) + "\n")
        for _ in range(garbage_lines):
            f.write("không phải json\n")


def test_dedupe_filter_and_output_format(tmp_path):
    raw = tmp_path / "raw" / "docs.jsonl"
    write_raw(raw, [
        ("A", make_detail("A", "Hết hiệu lực toàn bộ")),   # bản cũ của A
        ("B", make_detail("B", "Hết hiệu lực toàn bộ")),
        ("C", make_detail("C", html=None)),
        ("D", make_detail("D", html="<p>Không có cấu trúc</p>")),
        ("E", None),
        ("F", make_detail("F", status=None)),
        ("A", make_detail("A", "Còn hiệu lực")),           # bản mới nhất của A thắng
        ("G", make_detail("G", "Chưa có hiệu lực")),
    ], garbage_lines=1)

    out = tmp_path / "processed"
    report = run_preprocess(raw, out)

    processed = json.loads((out / "processed_data.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in processed] == ["A", "G"]
    assert processed[0]["effStatus"]["name"] == "Còn hiệu lực"
    assert processed[0]["metadata"][0]["title"] == "Chương I"
    assert processed[0]["documentContent"]["content"].startswith("<html>")

    errors = {e["doc_id"]: e["reason"] for e in json.loads((out / "error_items.json").read_text(encoding="utf-8"))}
    assert errors == {
        "B": "unsupported_effStatus: Hết hiệu lực toàn bộ",
        "C": "no_content",
        "D": "no_structure",
        "E": "detail_missing",
        "F": "missing_effStatus",
    }

    assert report.raw_lines == 9
    assert report.bad_lines == 1
    assert report.duplicates_removed == 1
    assert report.kept == 2
    assert report.rejected["unsupported_effStatus"] == 1
    saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert saved["kept"] == 2


def test_output_is_valid_json_when_nothing_kept(tmp_path):
    raw = tmp_path / "docs.jsonl"
    write_raw(raw, [("B", make_detail("B", "Ngưng hiệu lực"))])
    run_preprocess(raw, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "processed_data.json").read_text(encoding="utf-8")) == []


def test_report_counts_merged_rows(tmp_path):
    html = make_html([("prov-article", "a1", "Điều 1. X"), ("prov-article", "a1", "tiếp theo")])
    raw = tmp_path / "docs.jsonl"
    write_raw(raw, [("A", make_detail("A", html=html))])
    report = run_preprocess(raw, tmp_path / "out")
    assert report.merged_rows == 1
    assert report.docs_with_merged_rows == 1
    assert json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))["merged_rows"] == 1
