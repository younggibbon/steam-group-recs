"""HTTP слой: лимиты запросов, повторы с backoff и кеш ответов в SQLite.

Транспорт подменяемый: в проде это requests, в тестах фейковая функция,
поэтому весь сбор можно проверить без сети и без API ключа.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlparse

log = logging.getLogger(__name__)

# (url, params) -> (status_code, json или None)
Transport = Callable[[str, dict], tuple[int, Any]]

# Параметры, которые не должны попадать в ключ кеша (и в логи).
SECRET_PARAMS = {"key"}


class TransientError(Exception):
    """Сетевая ошибка, которую имеет смысл повторить."""


class HttpError(Exception):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


def requests_transport(timeout: float = 30.0) -> Transport:
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "steam-group-recs/0.1 (pet project)"

    def _get(url: str, params: dict) -> tuple[int, Any]:
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise TransientError(str(exc)) from exc
        try:
            data = resp.json()
        except ValueError:
            data = None
        return resp.status_code, data

    return _get


class RateLimiter:
    """Минимальный интервал между запросами к одному хосту."""

    def __init__(self, min_interval: float, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last = -float("inf")

    def wait(self) -> None:
        delay = self._last + self.min_interval - self._clock()
        if delay > 0:
            self._sleep(delay)
        self._last = self._clock()


class JsonCache:
    """Кеш ответов. Повторный запуск не тратит лимиты на то, что уже скачано."""

    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, status INTEGER, body TEXT, fetched_at REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> tuple[int, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status, body FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1]) if row[1] is not None else None

    def set(self, key: str, status: int, body: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                (key, status, json.dumps(body) if body is not None else None, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def cache_key(url: str, params: dict) -> str:
    public = {k: v for k, v in params.items() if k not in SECRET_PARAMS}
    return f"{url}?{urlencode(sorted(public.items()))}"


class HttpClient:
    # Статусы, которые означают «доступа нет» (закрытый профиль и т.п.).
    # Кешируем их, чтобы не переспрашивать при повторном запуске.
    CACHEABLE_ERRORS = {400, 401, 403, 404}
    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(
        self,
        transport: Transport,
        cache: JsonCache | None = None,
        limiters: dict[str, RateLimiter] | None = None,
        max_retries: int = 5,
        backoff: float = 2.0,
        backoff_429: float = 30.0,
        sleep=time.sleep,
    ):
        self.transport = transport
        self.cache = cache
        self.limiters = limiters or {}
        self.max_retries = max_retries
        self.backoff = backoff
        self.backoff_429 = backoff_429
        self._sleep = sleep
        self.network_calls = 0

    def get_json(self, url: str, params: dict | None = None) -> tuple[int, Any]:
        params = dict(params or {})
        key = cache_key(url, params)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        limiter = self.limiters.get(urlparse(url).netloc)
        for attempt in range(self.max_retries + 1):
            if limiter:
                limiter.wait()
            try:
                self.network_calls += 1
                status, data = self.transport(url, params)
            except TransientError as exc:
                status, data = None, None
                log.warning("Сетевая ошибка %s: %s", key, exc)

            if status is not None and status not in self.RETRYABLE:
                if self.cache is not None and (status == 200 or status in self.CACHEABLE_ERRORS):
                    self.cache.set(key, status, data)
                return status, data

            if attempt == self.max_retries:
                break
            base = self.backoff_429 if status == 429 else self.backoff
            delay = base * (2**attempt)
            log.info("Повтор %s через %.0f с (статус %s)", key, delay, status)
            self._sleep(delay)

        raise HttpError(status or 0, key)
