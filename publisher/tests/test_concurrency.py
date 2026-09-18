import asyncio

import pytest

from app.config import Settings
from app.telegram_client import TelegramPublisher


class DummyClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_input_entity(self, destination):
        return destination


class TrackingPublisher(TelegramPublisher):
    def __init__(self):
        super().__init__(
            Settings(
                api_id=1,
                api_hash="hash",
                session_string="session",
                publish_key="secret",
            )
        )
        self.active_sends = 0
        self.max_active_sends = 0

    def _client(self):
        return DummyClient()

    async def _send(self, client, peer, batch_id, body):
        self.active_sends += 1
        self.max_active_sends = max(self.max_active_sends, self.active_sends)
        await asyncio.sleep(0.02)
        self.active_sends -= 1
        return 100


@pytest.mark.asyncio
async def test_concurrent_publish_calls_are_serialized():
    publisher = TrackingPublisher()

    await asyncio.gather(
        publisher.publish("batch-1", "one"),
        publisher.publish("batch-2", "two"),
    )

    assert publisher.max_active_sends == 1
