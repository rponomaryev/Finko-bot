import importlib
import os

from fastapi.testclient import TestClient


os.environ.setdefault("TELEGRAM_API_ID", "1")
os.environ.setdefault("TELEGRAM_API_HASH", "hash")
os.environ.setdefault("TELEGRAM_SESSION_STRING", "session")
os.environ.setdefault("FINKO_PUBLISH_KEY", "secret")

main = importlib.import_module("app.main")


class FakePublisher:
    async def publish(self, batch_id: str, body: str) -> int:
        return 777


def test_health():
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_publish_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(main, "publisher", FakePublisher())
    monkeypatch.setattr(main, "PUBLISH_KEY", "secret")
    client = TestClient(main.app)
    response = client.post("/publish", json={"batch_id": "b1", "text": "Hello"})
    assert response.status_code == 401


def test_publish_returns_confirmed_message_id(monkeypatch):
    monkeypatch.setattr(main, "publisher", FakePublisher())
    monkeypatch.setattr(main, "PUBLISH_KEY", "secret")
    client = TestClient(main.app)
    response = client.post(
        "/publish",
        headers={"X-FINKO-PUBLISH-KEY": "secret"},
        json={"batch_id": "b1", "text": "Hello"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "batch_id": "b1",
        "chat": "@finkouz",
        "message_id": 777,
    }


def test_publish_rejects_oversized_text(monkeypatch):
    monkeypatch.setattr(main, "publisher", FakePublisher())
    monkeypatch.setattr(main, "PUBLISH_KEY", "secret")
    client = TestClient(main.app)
    response = client.post(
        "/publish",
        headers={"X-FINKO-PUBLISH-KEY": "secret"},
        json={"batch_id": "b1", "text": "x" * 3900},
    )
    assert response.status_code == 422


class FakeProbePublisher(FakePublisher):
    async def probe_custom_emojis(self):
        class Result:
            ok = True
            message_id = 888
            returned_custom_emoji_ids = (1, 2, 3, 4)
            error = None
        return Result()


def test_internal_probe_requires_token(monkeypatch):
    monkeypatch.setattr(main, "publisher", FakeProbePublisher())
    monkeypatch.setattr(main, "PROBE_TOKEN", "probe-secret")
    client = TestClient(main.app)
    response = client.get("/internal/probe/wrong")
    assert response.status_code == 404


def test_internal_probe_returns_entity_ids(monkeypatch):
    monkeypatch.setattr(main, "publisher", FakeProbePublisher())
    monkeypatch.setattr(main, "PROBE_TOKEN", "probe-secret")
    client = TestClient(main.app)
    response = client.get("/internal/probe/probe-secret")
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "message_id": 888,
        "returned_custom_emoji_ids": [1, 2, 3, 4],
        "error": None,
    }
