"""Tests for agent.anthropic_usage_logger — D3 structured observability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.anthropic_usage_logger import AnthropicUsageLogger
from agent.rate_limit_tracker import parse_anthropic_rate_limit_headers


# ── Helpers ───────────────────────────────────────────────────────────

def _headers_200():
    return {
        "request-id": "req_abc123",
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "47",
        "anthropic-ratelimit-requests-reset": "2026-04-22T01:00:00Z",
        "anthropic-ratelimit-input-tokens-limit": "40000",
        "anthropic-ratelimit-input-tokens-remaining": "38000",
        "anthropic-ratelimit-input-tokens-reset": "2026-04-22T01:00:00Z",
        "anthropic-ratelimit-tokens-limit": "800000",
        "anthropic-ratelimit-tokens-remaining": "450000",
        "anthropic-ratelimit-tokens-reset": "2026-04-22T02:00:00Z",
    }


def _headers_429():
    h = _headers_200()
    h["retry-after"] = "30"
    return h


# ── parse_anthropic_rate_limit_headers ────────────────────────────────

class TestParseAnthropicRateLimitHeaders:
    def test_parses_full_anthropic_header_set(self):
        state = parse_anthropic_rate_limit_headers(_headers_200())
        assert state is not None
        assert state.provider == "anthropic"
        assert state.requests_min.limit == 50
        assert state.requests_min.remaining == 47
        assert state.tokens_min.limit == 40000
        assert state.tokens_hour.limit == 800000

    def test_no_anthropic_headers_returns_none(self):
        assert parse_anthropic_rate_limit_headers({"x-ratelimit-limit-requests": "10"}) is None

    def test_missing_buckets_default_zero(self):
        state = parse_anthropic_rate_limit_headers({
            "anthropic-ratelimit-requests-limit": "10",
            "anthropic-ratelimit-requests-remaining": "9",
        })
        assert state is not None
        assert state.requests_min.limit == 10
        assert state.tokens_min.limit == 0

    def test_case_insensitive_header_lookup(self):
        state = parse_anthropic_rate_limit_headers({
            "Anthropic-Ratelimit-Requests-Limit": "5",
            "Anthropic-Ratelimit-Requests-Remaining": "4",
        })
        assert state is not None
        assert state.requests_min.limit == 5


# ── JSONL log format ──────────────────────────────────────────────────

class TestUsageLoggerRecord:
    def _mklogger(self, tmp_path: Path, alert_sink=None, now=None):
        log_path = tmp_path / "anthropic-usage.jsonl"
        kw = {"log_path": log_path}
        if alert_sink is not None:
            kw["alert_sink"] = alert_sink
        if now is not None:
            kw["now"] = now
        return AnthropicUsageLogger(**kw), log_path

    def test_writes_jsonl_line_with_core_fields(self, tmp_path):
        log, log_path = self._mklogger(tmp_path)
        log.record(
            headers=_headers_200(),
            model="claude-opus-4-7",
            status=200,
            latency_ms=423,
            input_tokens=1200,
            output_tokens=80,
        )
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["model"] == "claude-opus-4-7"
        assert rec["status"] == 200
        assert rec["latency_ms"] == 423
        assert rec["input_tokens"] == 1200
        assert rec["output_tokens"] == 80
        assert rec["request_id"] == "req_abc123"
        assert rec["rate_limit"]["requests_remaining"] == 47
        assert rec["rate_limit"]["tokens_pool_remaining"] == 450000

    def test_captures_rate_limit_headers_in_record(self, tmp_path):
        log, log_path = self._mklogger(tmp_path)
        log.record(headers=_headers_200(), model="claude-opus-4-7", status=200)
        rec = json.loads(log_path.read_text().splitlines()[0])
        captured = rec["headers"]
        assert captured["anthropic-ratelimit-requests-limit"] == "50"
        assert captured["anthropic-ratelimit-requests-remaining"] == "47"

    def test_redacts_authorization_header(self, tmp_path):
        """Authorization must never appear in captured headers."""
        log, log_path = self._mklogger(tmp_path)
        headers = dict(_headers_200())
        headers["authorization"] = "Bearer sk-ant-oat01-secret-token-abc"
        headers["x-api-key"] = "sk-ant-api03-another-secret"
        log.record(headers=headers, model="claude-opus-4-7", status=200)
        raw = log_path.read_text()
        assert "sk-ant-oat01-secret-token-abc" not in raw
        assert "sk-ant-api03-another-secret" not in raw
        assert "authorization" not in json.loads(raw.splitlines()[0])["headers"]

    def test_redacts_tokens_in_error_message(self, tmp_path):
        log, log_path = self._mklogger(tmp_path)
        log.record(
            headers={},
            model="claude-opus-4-7",
            status=401,
            error="Unauthorized: token sk-ant-oat01-abcdefghij-xyz is invalid",
        )
        raw = log_path.read_text()
        assert "sk-ant-oat01-abcdefghij-xyz" not in raw
        rec = json.loads(raw.splitlines()[0])
        assert rec["error"] is not None

    def test_no_anthropic_headers_no_rate_limit_subobject(self, tmp_path):
        log, log_path = self._mklogger(tmp_path)
        log.record(headers={}, model="claude-opus-4-7", status=500)
        rec = json.loads(log_path.read_text().splitlines()[0])
        assert "rate_limit" not in rec


# ── Sustained alert trigger ───────────────────────────────────────────

class TestSustainedAlertTrigger:
    def test_three_429s_in_window_fires_alert(self, tmp_path):
        fired = []
        t = [1_000_000.0]
        log = AnthropicUsageLogger(
            log_path=tmp_path / "u.jsonl",
            alert_sink=lambda msg: fired.append(msg),
            now=lambda: t[0],
        )
        for _ in range(3):
            log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
            t[0] += 60  # 60s apart -> 3 hits within 300s window

        assert len(fired) == 1
        assert "claude-opus-4-7" in fired[0]
        assert "5min" in fired[0]
        assert "3" in fired[0]

    def test_two_429s_below_threshold_no_alert(self, tmp_path):
        fired = []
        t = [1_000_000.0]
        log = AnthropicUsageLogger(
            log_path=tmp_path / "u.jsonl",
            alert_sink=lambda msg: fired.append(msg),
            now=lambda: t[0],
        )
        log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
        t[0] += 60
        log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
        assert fired == []

    def test_429s_spread_beyond_window_no_alert(self, tmp_path):
        """Three 429s over 10 minutes should not trip the 5-minute window."""
        fired = []
        t = [1_000_000.0]
        log = AnthropicUsageLogger(
            log_path=tmp_path / "u.jsonl",
            alert_sink=lambda msg: fired.append(msg),
            now=lambda: t[0],
        )
        log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
        t[0] += 301  # outside window
        log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
        t[0] += 301
        log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
        assert fired == []

    def test_alert_debounced_within_an_hour(self, tmp_path):
        fired = []
        t = [1_000_000.0]
        log = AnthropicUsageLogger(
            log_path=tmp_path / "u.jsonl",
            alert_sink=lambda msg: fired.append(msg),
            now=lambda: t[0],
        )
        for _ in range(3):
            log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
            t[0] += 60
        assert len(fired) == 1
        # 6 more 429s 30 minutes later — still within the 1h debounce
        t[0] += 1800
        for _ in range(6):
            log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
            t[0] += 30
        assert len(fired) == 1

    def test_alert_refires_after_debounce_window(self, tmp_path):
        fired = []
        t = [1_000_000.0]
        log = AnthropicUsageLogger(
            log_path=tmp_path / "u.jsonl",
            alert_sink=lambda msg: fired.append(msg),
            now=lambda: t[0],
        )
        for _ in range(3):
            log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
            t[0] += 60
        assert len(fired) == 1
        t[0] += 3700  # past 1h debounce
        for _ in range(3):
            log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
            t[0] += 60
        assert len(fired) == 2

    def test_alert_debounce_is_per_model(self, tmp_path):
        fired = []
        t = [1_000_000.0]
        log = AnthropicUsageLogger(
            log_path=tmp_path / "u.jsonl",
            alert_sink=lambda msg: fired.append(msg),
            now=lambda: t[0],
        )
        for _ in range(3):
            log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
            t[0] += 60
        for _ in range(3):
            log.record(headers=_headers_429(), model="claude-sonnet-4-6", status=429)
            t[0] += 60
        assert len(fired) == 2

    def test_non_429_status_does_not_count_toward_alert(self, tmp_path):
        fired = []
        t = [1_000_000.0]
        log = AnthropicUsageLogger(
            log_path=tmp_path / "u.jsonl",
            alert_sink=lambda msg: fired.append(msg),
            now=lambda: t[0],
        )
        for _ in range(3):
            log.record(headers=_headers_200(), model="claude-opus-4-7", status=200)
            t[0] += 60
        assert fired == []

    def test_alert_sink_failure_does_not_break_record(self, tmp_path):
        def bad_sink(msg):
            raise RuntimeError("boom")

        t = [1_000_000.0]
        log = AnthropicUsageLogger(
            log_path=tmp_path / "u.jsonl",
            alert_sink=bad_sink,
            now=lambda: t[0],
        )
        for _ in range(3):
            log.record(headers=_headers_429(), model="claude-opus-4-7", status=429)
            t[0] += 60
        # All three records should have been written despite sink failure
        lines = (tmp_path / "u.jsonl").read_text().splitlines()
        assert len(lines) == 3
