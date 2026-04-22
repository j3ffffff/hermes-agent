import json

from agent.memory_search import LocalMemorySearch
from tools.memory_tool import MemoryStore


class StubSessionDB:
    def __init__(self):
        self.calls = []

    def search_messages(self, query, limit=20, role_filter=None):
        self.calls.append((query, limit, role_filter))
        return [
            {
                "session_id": "s1",
                "content": "User said the repo uses pnpm and prefers concise updates.",
                "role": "user",
                "source": "cli",
                "session_started": 1700000000,
            },
            {
                "session_id": "s2",
                "content": "Assistant noted build command is npm run build.",
                "role": "assistant",
                "source": "cli",
                "session_started": 1700000100,
            },
        ]


def make_store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    store = MemoryStore(memory_char_limit=1000, user_char_limit=1000)
    store.load_from_disk()
    return store


class TestLocalMemorySearch:
    def test_search_durable_memory_returns_matches(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, monkeypatch)
        store.add("memory", "Build command is npm run build")
        store.add("user", "User prefers concise updates")

        search = LocalMemorySearch(memory_store=store)
        results = search.search_durable_memory("build concise")

        assert len(results) == 2
        assert any("npm run build" in item["content"] for item in results)
        assert any("concise updates" in item["content"] for item in results)

    def test_search_recent_sessions_uses_session_db(self):
        db = StubSessionDB()
        search = LocalMemorySearch(session_db=db)
        results = search.search_recent_sessions("build")

        assert len(results) == 2
        assert db.calls[0][0] == "build"

    def test_build_recall_context_dedupes_and_ranks(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, monkeypatch)
        store.add("memory", "Build command is npm run build")
        store.add("memory", "Build command is npm run build")
        store.add("user", "User prefers concise updates")

        db = StubSessionDB()
        search = LocalMemorySearch(memory_store=store, session_db=db)
        payload = search.build_recall_context("build concise", mode="full")

        assert payload["query"] == "build concise"
        assert payload["mode"] == "full"
        assert payload["counts"]["durable"] >= 1
        assert payload["counts"]["recent"] >= 1
        rendered = payload["rendered"]
        assert "npm run build" in rendered
        assert "concise updates" in rendered
        assert rendered.count("npm run build") == 1

    def test_build_recall_context_recent_mode_excludes_durable(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, monkeypatch)
        store.add("memory", "Build command is npm run build")

        db = StubSessionDB()
        search = LocalMemorySearch(memory_store=store, session_db=db)
        payload = search.build_recall_context("build", mode="recent")

        assert payload["counts"]["durable"] == 0
        assert payload["counts"]["recent"] >= 1
        assert "Recent session context" in payload["rendered"]

    def test_invalid_mode_raises(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, monkeypatch)
        search = LocalMemorySearch(memory_store=store)

        try:
            search.build_recall_context("test", mode="bogus")
        except ValueError as exc:
            assert "Invalid mode" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid mode")
