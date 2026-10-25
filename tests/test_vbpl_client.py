import pytest
import requests

from legal_rag.config import LIST_QUERY
from legal_rag.vbpl_client import VbplClient, VbplError

_INVALID = object()


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if self._body is _INVALID:
            raise ValueError("Expecting value")
        return self._body


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_client(*responses):
    session = FakeSession(*responses)
    return VbplClient(base_url="https://gw/doc", session_factory=lambda: session), session


def test_list_page_sends_query_and_parses_items():
    body = {"success": True, "data": {"total": 37673, "pageNumber": 2, "pageSize": 10,
                                      "items": [{"id": "1"}, {"id": "2"}]}}
    client, session = make_client(FakeResponse(200, body))
    page = client.list_page(2, 10)

    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "https://gw/doc/all")
    assert kwargs["json"] == {**LIST_QUERY, "pageSize": 10, "pageNumber": 2}
    assert page.total == 37673
    assert [it["id"] for it in page.items] == ["1", "2"]


def test_envelope_without_data_is_api_error():
    # Hình dạng thật khi gateway lỗi: HTTP 200, không có "data", có "message"
    body = {"success": None, "statusCode": None, "message": "Đã có lỗi xảy ra (Elasticsearch exception)"}
    client, _ = make_client(FakeResponse(200, body))
    with pytest.raises(VbplError) as exc:
        client.list_page(1, 10)
    assert exc.value.reason == "api_error"
    assert "Elasticsearch" in exc.value.detail


def test_http_error_status_is_mapped():
    client, _ = make_client(FakeResponse(503, None))
    with pytest.raises(VbplError) as exc:
        client.get_detail("1")
    assert exc.value.reason == "http_503"


def test_timeout_and_connection_errors_are_mapped():
    client, _ = make_client(requests.ConnectTimeout("t"), requests.ConnectionError("c"))
    with pytest.raises(VbplError) as exc:
        client.get_detail("1")
    assert exc.value.reason == "timeout"
    with pytest.raises(VbplError) as exc:
        client.get_detail("1")
    assert exc.value.reason == "connection"


def test_detail_null_is_error_not_written():
    client, _ = make_client(FakeResponse(200, {"data": None}))
    with pytest.raises(VbplError) as exc:
        client.get_detail("1")
    assert exc.value.reason == "detail_null"


def test_bad_json_is_mapped():
    client, _ = make_client(FakeResponse(200, _INVALID))
    with pytest.raises(VbplError) as exc:
        client.get_detail("1")
    assert exc.value.reason == "bad_json"


def test_get_detail_returns_data():
    client, session = make_client(FakeResponse(200, {"data": {"id": "187835", "docNum": "135/2026/NĐ-CP"}}))
    detail = client.get_detail("187835")
    assert detail["docNum"] == "135/2026/NĐ-CP"
    assert session.calls[0][:2] == ("GET", "https://gw/doc/187835")
