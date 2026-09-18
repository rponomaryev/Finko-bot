import asyncio

from app.config import Settings
from app.telegram_client import TelegramPublisher, random_id_for


class DummyClient:
    def __init__(self):
        self.requested = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_input_entity(self, destination):
        self.requested.append(destination)
        return destination


class TrackingPublisher(TelegramPublisher):
    def __init__(self):
        super().__init__(
            Settings(
                api_id=1,
                api_hash="h",
                session_string="s",
                publish_key="k",
            )
        )
        self.client = DummyClient()
        self.sent = []

    def _client(self):
        return self.client

    async def _send(self, client, peer, batch_id, body):
        self.sent.append((peer, batch_id, body))
        return 42


def test_preview_uses_review_destination_and_distinct_idempotency_scope():
    async def run_case():
        publisher = TrackingPublisher()
        message_id = await publisher.preview("batch-1", "body")

        assert message_id == 42
        assert publisher.client.requested == [-5547276399]
        assert publisher.sent == [
            (-5547276399, "preview:batch-1", "body")
        ]

    asyncio.run(run_case())


def test_channel_publish_keeps_channel_destination_and_own_scope():
    async def run_case():
        publisher = TrackingPublisher()
        message_id = await publisher.publish("batch-1", "body")

        assert message_id == 42
        assert publisher.client.requested == ["@finkouz"]
        assert publisher.sent == [
            ("@finkouz", "channel:batch-1", "body")
        ]

    asyncio.run(run_case())


def test_preview_and_channel_random_ids_do_not_collide():
    assert random_id_for("preview:batch-1") != random_id_for("channel:batch-1")
