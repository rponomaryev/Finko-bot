from types import SimpleNamespace

from app.telegram_client import TelegramPublisher


def test_message_id_from_update_short_sent_message_shape():
    result = SimpleNamespace(id=173)
    assert TelegramPublisher._message_id_from_updates(result) == 173


def test_message_id_from_updates_message_shape():
    result = SimpleNamespace(
        updates=[SimpleNamespace(message=SimpleNamespace(id=42))]
    )
    assert TelegramPublisher._message_id_from_updates(result) == 42
