import unicodedata

from legal_rag.structure import build_provision_tree, node_title
from tests.conftest import SAMPLE_HTML, make_html


def _flatten(nodes, depth=0):
    for node in nodes:
        yield depth, node
        yield from _flatten(node["children"], depth + 1)


def test_tree_follows_legal_hierarchy():
    tree, stats = build_provision_tree(SAMPLE_HTML)
    outline = [(depth, n["title"]) for depth, n in _flatten(tree)]
    assert outline == [
        (0, "Chương I"),
        (1, "Điều 1"),
        (0, "Chương II"),
        (1, "Điều 17"),
        (2, "Khoản 1"),
        (2, "Khoản 2"),
        (3, "Điểm a"),
        (3, "Điểm b"),
    ]
    assert stats.levels == {"Chapter": 2, "Article": 2, "Clause": 2, "Point": 2}


def test_leaf_flags_and_order_index():
    tree, _ = build_provision_tree(SAMPLE_HTML)
    nodes = {n["id"]: n for _, n in _flatten(tree)}
    # Điều không có Khoản là lá; Khoản có Điểm thì không phải lá
    assert nodes["a1"]["isLeaf"] is True
    assert nodes["a17"]["isLeaf"] is False
    assert nodes["k1"]["isLeaf"] is True
    assert nodes["k2"]["isLeaf"] is False
    assert nodes["pb"]["isLeaf"] is True
    assert [n["orderIndex"] for n in nodes["k2"]["children"]] == [1, 2]


def test_node_schema_matches_phase3_metadata():
    tree, _ = build_provision_tree(SAMPLE_HTML)
    for _, node in _flatten(tree):
        assert set(node) == {"id", "title", "level", "ptype", "orderIndex", "children", "isLeaf"}


def test_content_rows_and_missing_ids_do_not_create_nodes():
    html = make_html([
        ("prov-article", "a1", "Điều 1. Phạm vi"),
        ("prov-content", None, "đoạn dẫn"),
        ("prov-clause", None, "1. Khoản không có id"),
        ("prov-annex", "x1", "Phụ lục I"),
    ])
    tree, stats = build_provision_tree(html)
    assert [n["id"] for _, n in _flatten(tree)] == ["a1"]
    assert stats.skipped_no_id == 1
    assert stats.unknown_classes == {"prov-annex": 1}


def test_clause_directly_under_chapter_is_allowed():
    html = make_html([("prov-chapter", "c1", "Chương I"), ("prov-clause", "k1", "1. Nội dung")])
    tree, _ = build_provision_tree(html)
    assert tree[0]["children"][0]["title"] == "Khoản 1"


def test_decomposed_vietnamese_is_normalized():
    nfd = unicodedata.normalize("NFD", "Điều 3. Giải thích từ ngữ")
    tree, _ = build_provision_tree(make_html([("prov-article", "a3", nfd)]))
    assert tree[0]["title"] == unicodedata.normalize("NFC", "Điều 3")


def test_continuation_row_with_same_id_is_merged_not_a_new_node():
    # Hình dạng thật trong Bộ luật Dân sự: tiêu đề Mục tách 2 <p> cùng id
    html = make_html([
        ("prov-section", "s1", "Mục 1"),
        ("prov-section", "s1", "NĂNG LỰC PHÁP LUẬT DÂN SỰ"),
        ("prov-article", "a16", "Điều 16. Năng lực pháp luật dân sự của cá nhân"),
    ])
    tree, stats = build_provision_tree(html)
    assert stats.merged_rows == 1
    assert len(tree) == 1
    assert tree[0]["title"] == "Mục 1"
    assert tree[0]["isLeaf"] is False
    assert tree[0]["children"][0]["title"] == "Điều 16"


def test_empty_or_unstructured_html_gives_empty_tree():
    assert build_provision_tree("")[0] == []
    assert build_provision_tree("<p>Văn bản không có đánh dấu cấu trúc</p>")[0] == []


def test_titles_fall_back_to_text_prefix():
    assert node_title("Clause", "Trường hợp người được thử nghiệm là người chưa thành niên") == "Khoản không đánh số"
    assert node_title("Article", "Điều khoản chuyển tiếp không số") == "Điều khoản chuyển tiếp không số"
    assert node_title("Point", "đ) Kỹ sư SCADA") == "Điểm đ"
    assert node_title("Section", "Mục 2. QUYỀN") == "Mục 2"
    assert node_title("Part", "Phần thứ nhất NHỮNG QUY ĐỊNH CHUNG") == "Phần thứ nhất"
