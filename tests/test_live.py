"""Smoke test gọi API thật. Mặc định bị bỏ qua; chạy bằng: RUN_LIVE=1 pytest -m live"""

import os

import pytest

from legal_rag.structure import build_provision_tree
from legal_rag.vbpl_client import VbplClient

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="đặt RUN_LIVE=1 để gọi API thật"),
]


def test_list_and_detail_shapes():
    client = VbplClient()
    listing = client.list_page(1, 2)
    assert listing.total and listing.total > 0
    assert len(listing.items) == 2
    assert {"id", "docNum", "effStatus"} <= set(listing.items[0])

    detail = client.get_detail(str(listing.items[0]["id"]))
    assert {"id", "docNum", "title", "effStatus", "documentContent"} <= set(detail)
    tree, stats = build_provision_tree(detail["documentContent"]["content"])
    assert tree, "HTML không còn class prov-*: cần xem lại structure.py"
    assert stats.levels["Article"] > 0
