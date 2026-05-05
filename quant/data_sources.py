"""External data access helpers with retry, rate limits and SQLite TTL cache."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any

import pandas as pd
import requests
import yfinance as yf


HTTP_TIMEOUT_SEC = float(os.environ.get("HTTP_TIMEOUT_SEC", "20"))
HTTP_RETRIES = int(os.environ.get("HTTP_RETRIES", "3"))
HTTP_BACKOFF_SEC = float(os.environ.get("HTTP_BACKOFF_SEC", "1.5"))

EXTERNAL_CACHE_PATH = os.environ.get("EXTERNAL_CACHE_PATH", "config/external_cache.sqlite3")
EXTERNAL_CACHE_TTL_HOURS = float(os.environ.get("EXTERNAL_CACHE_TTL_HOURS", "12"))

GOODINFO_MIN_INTERVAL_SEC = float(os.environ.get("GOODINFO_MIN_INTERVAL_SEC", "3.0"))
FINMIND_MIN_INTERVAL_SEC = float(os.environ.get("FINMIND_MIN_INTERVAL_SEC", "2.0"))

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_session = requests.Session()
_rate_lock = threading.Lock()
_last_call_at: dict[str, float] = {}


def _json_default(value: Any) -> str:
    return str(value)


class SQLiteTTLCache:
    """Tiny JSON TTL cache for external API responses."""

    def __init__(self, path: str = EXTERNAL_CACHE_PATH):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_cache (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_seconds REAL NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )

    def get(self, namespace: str, cache_key: str) -> Any | None:
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT value_json, created_at, ttl_seconds
                FROM external_cache
                WHERE namespace = ? AND cache_key = ?
                """,
                (namespace, cache_key),
            ).fetchone()
            if not row:
                return None
            value_json, created_at, ttl_seconds = row
            if now - float(created_at) > float(ttl_seconds):
                conn.execute(
                    "DELETE FROM external_cache WHERE namespace = ? AND cache_key = ?",
                    (namespace, cache_key),
                )
                return None
            return json.loads(value_json)

    def set(self, namespace: str, cache_key: str, value: Any, ttl_hours: float | None = None) -> None:
        ttl = float(EXTERNAL_CACHE_TTL_HOURS if ttl_hours is None else ttl_hours) * 3600
        payload = json.dumps(value, ensure_ascii=False, default=_json_default)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO external_cache(namespace, cache_key, value_json, created_at, ttl_seconds)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key)
                DO UPDATE SET value_json = excluded.value_json,
                              created_at = excluded.created_at,
                              ttl_seconds = excluded.ttl_seconds
                """,
                (namespace, cache_key, payload, time.time(), ttl),
            )

    def clear_expired(self) -> int:
        now = time.time()
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT namespace, cache_key, created_at, ttl_seconds FROM external_cache").fetchall()
            expired = [(ns, key) for ns, key, created_at, ttl in rows if now - float(created_at) > float(ttl)]
            for ns, key in expired:
                conn.execute("DELETE FROM external_cache WHERE namespace = ? AND cache_key = ?", (ns, key))
            return len(expired)


cache = SQLiteTTLCache()


def rate_limited(key: str, min_interval_sec: float) -> None:
    if min_interval_sec <= 0:
        return
    with _rate_lock:
        now = time.time()
        wait = min_interval_sec - (now - _last_call_at.get(key, 0))
        if wait > 0:
            time.sleep(wait)
        _last_call_at[key] = time.time()


def request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = HTTP_TIMEOUT_SEC,
    retries: int = HTTP_RETRIES,
    backoff_sec: float = HTTP_BACKOFF_SEC,
    **kwargs: Any,
) -> requests.Response:
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            response = _session.request(method, url, headers=merged_headers, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff_sec * (2**attempt))
    raise RuntimeError(f"external request failed after {retries} attempts: {url}") from last_error


def get_text(
    url: str,
    *,
    namespace: str | None = None,
    cache_key: str | None = None,
    ttl_hours: float | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = HTTP_TIMEOUT_SEC,
) -> str:
    if namespace and cache_key:
        cached = cache.get(namespace, cache_key)
        if cached is not None:
            return str(cached)
    response = request_with_retry("GET", url, headers=headers, timeout=timeout)
    text = response.text
    if namespace and cache_key:
        cache.set(namespace, cache_key, text, ttl_hours)
    return text


def get_json(
    url: str,
    *,
    namespace: str | None = None,
    cache_key: str | None = None,
    ttl_hours: float | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = HTTP_TIMEOUT_SEC,
) -> Any:
    if namespace and cache_key:
        cached = cache.get(namespace, cache_key)
        if cached is not None:
            return cached
    payload = request_with_retry("GET", url, headers=headers, timeout=timeout).json()
    if namespace and cache_key:
        cache.set(namespace, cache_key, payload, ttl_hours)
    return payload


def download_yfinance(ticker: str, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(max(1, HTTP_RETRIES)):
        try:
            return yf.download(ticker, progress=False, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < HTTP_RETRIES - 1:
                time.sleep(HTTP_BACKOFF_SEC * (2**attempt))
    raise RuntimeError(f"yfinance download failed after {HTTP_RETRIES} attempts: {ticker}") from last_error


def get_yahoo_info(ticker: str, ttl_hours: float = 12) -> dict[str, Any]:
    key = ticker.strip().upper()
    cached = cache.get("yahoo_info", key)
    if cached is not None:
        return dict(cached)

    last_error: Exception | None = None
    for attempt in range(max(1, HTTP_RETRIES)):
        try:
            info = yf.Ticker(ticker).info or {}
            cache.set("yahoo_info", key, info, ttl_hours)
            return dict(info)
        except Exception as exc:
            last_error = exc
            if attempt < HTTP_RETRIES - 1:
                time.sleep(HTTP_BACKOFF_SEC * (2**attempt))
    raise RuntimeError(f"Yahoo info failed after {HTTP_RETRIES} attempts: {ticker}") from last_error


def fetch_goodinfo_pages(ticker_num: str, ttl_hours: float = 24) -> tuple[str, str]:
    key = str(ticker_num).strip()
    cached = cache.get("goodinfo_pages", key)
    if cached is not None:
        return str(cached.get("main_html", "")), str(cached.get("chip_html", ""))

    rate_limited("goodinfo", GOODINFO_MIN_INTERVAL_SEC)
    from playwright.sync_api import sync_playwright

    url_main = f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={key}"
    url_chip = f"https://goodinfo.tw/tw/ShowBuySaleChart.asp?STOCK_ID={key}&CHT_CAT=DATE"
    main_html = ""
    chip_html = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"])
        page = context.new_page()
        try:
            page.goto(url_main, wait_until="domcontentloaded", timeout=int(HTTP_TIMEOUT_SEC * 1000))
            page.wait_for_timeout(1500)
            main_html = page.content()
        except Exception:
            main_html = ""
        try:
            rate_limited("goodinfo", GOODINFO_MIN_INTERVAL_SEC)
            page.goto(url_chip, wait_until="domcontentloaded", timeout=int(HTTP_TIMEOUT_SEC * 1000))
            page.wait_for_timeout(1500)
            chip_html = page.content()
        except Exception:
            chip_html = ""
        browser.close()

    cache.set("goodinfo_pages", key, {"main_html": main_html, "chip_html": chip_html}, ttl_hours)
    return main_html, chip_html


def fetch_finmind_institutional_investors(
    dl_client: Any,
    stock_id: str,
    start_date: str,
    end_date: str,
    ttl_hours: float = 6,
) -> pd.DataFrame:
    key = f"{stock_id}:{start_date}:{end_date}"
    cached = cache.get("finmind_institutional", key)
    if cached is not None:
        return pd.DataFrame(cached)

    if dl_client is None:
        return pd.DataFrame()

    rate_limited("finmind", FINMIND_MIN_INTERVAL_SEC)
    last_error: Exception | None = None
    for attempt in range(max(1, HTTP_RETRIES)):
        try:
            df = dl_client.taiwan_stock_institutional_investors(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )
            if df is None:
                return pd.DataFrame()
            records = df.to_dict(orient="records")
            cache.set("finmind_institutional", key, records, ttl_hours)
            return df
        except Exception as exc:
            last_error = exc
            if attempt < HTTP_RETRIES - 1:
                time.sleep(HTTP_BACKOFF_SEC * (2**attempt))
    raise RuntimeError(f"FinMind institutional data failed after {HTTP_RETRIES} attempts: {stock_id}") from last_error

