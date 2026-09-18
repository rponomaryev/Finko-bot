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


class FakeTwoStagePublisher(FakePublisher):
    async def preview(self, batch_id: str, body: str) -> int:
        return 555


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


def test_preview_returns_review_group_message_id(monkeypatch):
    monkeypatch.setattr(main, "publisher", FakeTwoStagePublisher())
    monkeypatch.setattr(main, "PUBLISH_KEY", "secret")
    client = TestClient(main.app)
    response = client.post(
        "/preview",
        headers={"X-FINKO-PUBLISH-KEY": "secret"},
        json={"batch_id": "b1", "text": "Hello"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "batch_id": "b1",
        "chat": "-5547276399",
        "message_id": 555,
    }
