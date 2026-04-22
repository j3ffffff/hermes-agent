"""Local memory search helpers for builtin memory + recent session recall.

This module is intentionally local-first and lightweight. It searches:
- built-in durable memory (MEMORY.md / USER.md via MemoryStore)
- recent session history (via SessionDB.search_messages when available)

It returns compact, deduped recall payloads suitable for later prompt injection,
dreaming, or active-memory style prefetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.memory_tool import MemoryStore

_VALID_MODES = {"recent", "durable", "full"}


@dataclass
class RecallItem:
    source: str
    content: str
    score: int
    metadata: Dict[str, Any]


class LocalMemorySearch:
    def __init__(self, memory_store: Optional[MemoryStore] = None, session_db: Any = None):
        self.memory_store = memory_store
        self.session_db = session_db

    def search_durable_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.memory_store or not query.strip():
            return []

        query_terms = self._query_terms(query)
        results: List[RecallItem] = []

        for target, entries in (("memory", self.memory_store.memory_entries), ("user", self.memory_store.user_entries)):
            for entry in entries:
                score = self._score_text(entry, query_terms)
                if score <= 0:
                    continue
                results.append(
                    RecallItem(
                        source=target,
                        content=entry,
                        score=score,
                        metadata={"target": target},
                    )
                )

        return [self._to_dict(item) for item in self._rank_and_dedupe(results)[:limit]]

    def search_recent_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.session_db or not query.strip():
            return []

        try:
            rows = self.session_db.search_messages(query=query, limit=limit * 3, role_filter=None)
        except TypeError:
            rows = self.session_db.search_messages(query, limit=limit * 3)

        results: List[RecallItem] = []
        query_terms = self._query_terms(query)
        for row in rows or []:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            score = self._score_text(content, query_terms)
            if score <= 0:
                score = 1
            results.append(
                RecallItem(
                    source="recent_session",
                    content=content,
                    score=score,
                    metadata={
                        "session_id": row.get("session_id"),
                        "role": row.get("role"),
                        "source": row.get("source"),
                        "session_started": row.get("session_started"),
                    },
                )
            )

        return [self._to_dict(item) for item in self._rank_and_dedupe(results)[:limit]]

    def build_recall_context(self, query: str, mode: str = "recent", durable_limit: int = 5, recent_limit: int = 5) -> Dict[str, Any]:
        mode = mode.strip().lower()
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of: {', '.join(sorted(_VALID_MODES))}")

        durable: List[Dict[str, Any]] = []
        recent: List[Dict[str, Any]] = []

        if mode in {"durable", "full"}:
            durable = self.search_durable_memory(query, limit=durable_limit)
        if mode in {"recent", "full"}:
            recent = self.search_recent_sessions(query, limit=recent_limit)

        rendered = self._render_context(durable, recent)
        return {
            "query": query,
            "mode": mode,
            "durable": durable,
            "recent": recent,
            "counts": {
                "durable": len(durable),
                "recent": len(recent),
            },
            "rendered": rendered,
        }

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        return [term.casefold() for term in query.split() if term.strip()]

    @staticmethod
    def _score_text(text: str, query_terms: List[str]) -> int:
        lowered = text.casefold()
        score = 0
        for term in query_terms:
            if term in lowered:
                score += 1
        return score

    @staticmethod
    def _canonicalize(text: str) -> str:
        return " ".join(text.casefold().split())

    @classmethod
    def _is_redundant_with_seen(cls, text: str, seen: set[str]) -> bool:
        canonical = cls._canonicalize(text)
        canonical_terms = set(canonical.split())
        for existing in seen:
            if canonical == existing or canonical in existing or existing in canonical:
                return True
            existing_terms = set(existing.split())
            if canonical_terms and existing_terms and len(canonical_terms & existing_terms) >= 3:
                return True
        return False

    def _rank_and_dedupe(self, items: List[RecallItem]) -> List[RecallItem]:
        deduped: List[RecallItem] = []
        seen = set()
        for item in sorted(items, key=lambda x: (x.score, x.source == "user"), reverse=True):
            key = self._canonicalize(item.content)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _to_dict(item: RecallItem) -> Dict[str, Any]:
        return {
            "source": item.source,
            "content": item.content,
            "score": item.score,
            "metadata": item.metadata,
        }

    @classmethod
    def _render_context(cls, durable: List[Dict[str, Any]], recent: List[Dict[str, Any]]) -> str:
        sections: List[str] = []
        seen = set()

        if durable:
            durable_lines = []
            for item in durable:
                key = cls._canonicalize(item["content"])
                if cls._is_redundant_with_seen(item["content"], seen):
                    continue
                seen.add(key)
                durable_lines.append(f"- [{item['source']}] {item['content']}")
            if durable_lines:
                sections.append("## Durable memory\n" + "\n".join(durable_lines))

        if recent:
            recent_lines = []
            for item in recent:
                key = cls._canonicalize(item["content"])
                if cls._is_redundant_with_seen(item["content"], seen):
                    continue
                seen.add(key)
                recent_lines.append(f"- {item['content']}")
            if recent_lines:
                sections.append("## Recent session context\n" + "\n".join(recent_lines))

        return "\n\n".join(sections)
