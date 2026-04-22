"""Structured observability for Anthropic API requests.

Appends a JSONL record per Anthropic call to
``~/.hermes/logs/anthropic-usage.jsonl`` and fires a Telegram alert when a
sustained burst of 429s is detected (3+ within a rolling 5-minute window).

The alert is debounced per model: once fired for a given model it will not
re-fire for an hour.

Secrets (tokens, API keys, bearer auth) are redacted from captured fields
via :mod:`agent.redact`. Rate-limit header values themselves are not
sensitive, but error bodies occasionally echo the caller's auth header —
hence the blanket redact.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional

from agent.redact import redact_sensitive_text
from agent.rate_limit_tracker import parse_anthropic_rate_limit_headers

logger = logging.getLogger(__name__)

# Header keys we copy into each JSONL record so operators can spot per-bucket
# pressure without re-parsing. Lowercased for case-insensitive lookup.
_CAPTURED_HEADERS = (
    "request-id",
    "retry-after",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-input-tokens-limit",
    "anthropic-ratelimit-input-tokens-remaining",
    "anthropic-ratelimit-input-tokens-reset",
    "anthropic-ratelimit-output-tokens-limit",
    "anthropic-ratelimit-output-tokens-remaining",
    "anthropic-ratelimit-output-tokens-reset",
    "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-tokens-remaining",
    "anthropic-ratelimit-tokens-reset",
)

# Sustained-alert tuning
_ALERT_WINDOW_SECONDS = 300          # rolling window for counting 429s
_ALERT_THRESHOLD = 3                 # 3+ 429s in window -> fire
_ALERT_DEBOUNCE_SECONDS = 3600       # don't re-fire for same model within 1h


def _default_log_path() -> Path:
    override = os.environ.get("HERMES_ANTHROPIC_USAGE_LOG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes" / "logs" / "anthropic-usage.jsonl"


def _redact_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Copy only the rate-limit-relevant headers. Never copy Authorization."""
    lowered = {k.lower(): v for k, v in headers.items()}
    return {k: str(lowered[k]) for k in _CAPTURED_HEADERS if k in lowered}


class AnthropicUsageLogger:
    """Per-process singleton that writes JSONL records and fires alerts."""

    def __init__(
        self,
        log_path: Optional[Path] = None,
        *,
        alert_sink=None,
        now=time.time,
    ) -> None:
        self._log_path = log_path or _default_log_path()
        self._lock = threading.Lock()
        self._window: Dict[str, Deque[float]] = defaultdict(deque)
        self._last_alert_at: Dict[str, float] = {}
        # Injection seam for tests
        self._alert_sink = alert_sink or _send_telegram_alert
        self._now = now

    def record(
        self,
        *,
        headers: Optional[Mapping[str, str]] = None,
        model: Optional[str] = None,
        status: Optional[int] = None,
        latency_ms: Optional[int] = None,
        request_id: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Append one record; fire sustained-alert if threshold tripped."""
        hdrs = dict(headers or {})
        rate_state = parse_anthropic_rate_limit_headers(hdrs)
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id or (
                {k.lower(): v for k, v in hdrs.items()}.get("request-id")
            ),
            "model": model,
            "status": status,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "headers": _redact_headers(hdrs),
            "error": redact_sensitive_text(error) if error else None,
        }
        if rate_state is not None:
            record["rate_limit"] = {
                "requests_remaining": rate_state.requests_min.remaining,
                "requests_limit": rate_state.requests_min.limit,
                "input_tokens_remaining": rate_state.tokens_min.remaining,
                "tokens_pool_remaining": rate_state.tokens_hour.remaining,
                "requests_reset_sec": rate_state.requests_min.remaining_seconds_now,
            }

        self._write(record)

        if status == 429:
            self._check_sustained_alert(model or "unknown")

    # ── internals ─────────────────────────────────────────────────────

    def _write(self, record: Dict[str, Any]) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            logger.debug("anthropic-usage.jsonl write failed: %s", e)

    def _check_sustained_alert(self, model: str) -> None:
        now = self._now()
        with self._lock:
            window = self._window[model]
            window.append(now)
            cutoff = now - _ALERT_WINDOW_SECONDS
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) < _ALERT_THRESHOLD:
                return
            last = self._last_alert_at.get(model, 0.0)
            if now - last < _ALERT_DEBOUNCE_SECONDS:
                return
            self._last_alert_at[model] = now
            count = len(window)

        msg = (
            f"Hermes hit {count} 429s in last 5min on model {model}. "
            "Consider switching alias or waiting for reset."
        )
        try:
            self._alert_sink(msg)
        except Exception as e:  # never let alert failure break the request path
            logger.debug("sustained-alert sink failed: %s", e)


def _send_telegram_alert(message: str) -> None:
    """Fire a Telegram message to the configured home channel.

    No-op when ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_HOME_CHANNEL`` is missing
    — the alert is best-effort and must never block the API path.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_HOME_CHANNEL")
    if not token or not chat_id:
        logger.info("sustained-alert (telegram unavailable): %s", message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.debug("telegram alert send failed: %s", e)


# ── module-level singleton ────────────────────────────────────────────

_default_logger: Optional[AnthropicUsageLogger] = None
_default_logger_lock = threading.Lock()


def get_default_logger() -> AnthropicUsageLogger:
    global _default_logger
    if _default_logger is None:
        with _default_logger_lock:
            if _default_logger is None:
                _default_logger = AnthropicUsageLogger()
    return _default_logger


def record_anthropic_call(**kwargs) -> None:
    """Convenience wrapper to record against the per-process singleton."""
    get_default_logger().record(**kwargs)
