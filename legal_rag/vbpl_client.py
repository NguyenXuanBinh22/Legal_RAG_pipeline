"""Client cho API công khai của VBPL gateway (danh sách + chi tiết văn bản)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import GATEWAY_BASE, LIST_QUERY, USER_AGENT

RETRY_STATUSES = (429, 500, 502, 503, 504)


class VbplError(Exception):
    """Lỗi khi gọi API. `reason` là mã ngắn, dùng để thống kê và ghi checkpoint."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class ListPage:
    page_number: int
    page_size: int
    total: int | None
    items: list[dict]


def make_session(retries: int, backoff: float, user_agent: str) -> requests.Session:
    """Session có retry ở MỘT tầng duy nhất (không chồng thêm vòng retry thủ công)."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=frozenset({"GET", "POST"}),  # cả 2 POST đều chỉ đọc
        respect_retry_after_header=True,
        raise_on_status=False,  # hết lượt retry thì trả response để map thành VbplError
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return session


class VbplClient:
    def __init__(
        self,
        base_url: str = GATEWAY_BASE,
        query: dict | None = None,
        timeout: tuple[float, float] = (5.0, 30.0),
        retries: int = 3,
        backoff: float = 1.0,
        delay: float = 0.0,
        user_agent: str = USER_AGENT,
        session_factory: Callable[[], Any] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.query = dict(LIST_QUERY if query is None else query)
        self.timeout = timeout
        self.delay = delay
        self._session_factory = session_factory or (lambda: make_session(retries, backoff, user_agent))
        # requests.Session không đảm bảo thread-safe → mỗi thread worker một session
        self._local = threading.local()

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = self._session_factory()
        return session

    def _call(self, method: str, url: str, **kwargs) -> dict:
        if self.delay:
            time.sleep(self.delay)
        try:
            resp = self._session().request(method, url, timeout=self.timeout, **kwargs)
        except requests.Timeout as e:
            raise VbplError("timeout", str(e)) from e
        except requests.ConnectionError as e:
            raise VbplError("connection", str(e)) from e
        except requests.RequestException as e:
            raise VbplError("request", str(e)) from e

        if resp.status_code != 200:
            raise VbplError(f"http_{resp.status_code}")
        try:
            body = resp.json()
        except ValueError as e:
            raise VbplError("bad_json", str(e)) from e
        # Khi lỗi, API vẫn trả 200 nhưng thiếu key "data" và có "message"
        if not isinstance(body, dict) or "data" not in body:
            message = body.get("message") if isinstance(body, dict) else ""
            raise VbplError("api_error", str(message or "")[:200])
        return body

    def list_page(self, page_number: int, page_size: int) -> ListPage:
        body = self._call(
            "POST",
            f"{self.base_url}/all",
            json={**self.query, "pageSize": page_size, "pageNumber": page_number},
        )
        data = body["data"] or {}
        items = data.get("items")
        if not isinstance(items, list):
            raise VbplError("api_error", "thiếu data.items")
        return ListPage(page_number, page_size, data.get("total"), items)

    def get_detail(self, doc_id: str) -> dict:
        body = self._call("GET", f"{self.base_url}/{doc_id}")
        data = body["data"]
        if not isinstance(data, dict) or not data:
            raise VbplError("detail_null")
        return data
