from dataclasses import dataclass
import asyncio
import hashlib

from .config import Settings
from .footer import EntitySpec, build_message


EXPECTED_CUSTOM_EMOJI_IDS = (
    5319160079465857105,
    5370722600668382252,
    5373130604147654226,
    5785209342986817408,
)


def random_id_for(batch_id: str) -> int:
    raw = hashlib.blake2b(
        batch_id.encode("utf-8"),
        digest_size=8,
        person=b"FINKOPUB",
    ).digest()
    unsigned = int.from_bytes(raw, "big", signed=False)
    if unsigned >= (1 << 63):
        return unsigned - (1 << 64)
    return unsigned


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    message_id: int | None
    returned_custom_emoji_ids: tuple[int, ...]
    error: str | None = None


def _telethon_entities(specs: list[EntitySpec]):
    from telethon.tl import types

    result = []
    for spec in specs:
        if spec.kind == "custom_emoji":
            result.append(
                types.MessageEntityCustomEmoji(
                    offset=spec.offset,
                    length=spec.length,
                    document_id=int(spec.document_id),
                )
            )
        elif spec.kind == "text_url":
            result.append(
                types.MessageEntityTextUrl(
                    offset=spec.offset,
                    length=spec.length,
                    url=str(spec.url),
                )
            )
        else:
            raise ValueError(f"unsupported_entity_kind:{spec.kind}")
    return result


class TelegramPublisher:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._publish_lock = asyncio.Lock()

    def _client(self):
        from telethon import TelegramClient
        from telethon.sessions import StringSession

        return TelegramClient(
            StringSession(self.settings.session_string),
            self.settings.api_id,
            self.settings.api_hash,
        )

    @staticmethod
    def _message_id_from_updates(result) -> int:
        for update in getattr(result, "updates", ()):
            message = getattr(update, "message", None)
            message_id = getattr(message, "id", None)
            if message_id is not None:
                return int(message_id)
        raise RuntimeError("telegram_message_id_missing")

    async def _send(self, client, peer, batch_id: str, body: str) -> int:
        from telethon.tl import functions

        text, specs = build_message(body)
        entities = _telethon_entities(specs)
        result = await client(
            functions.messages.SendMessageRequest(
                peer=peer,
                message=text,
                random_id=random_id_for(batch_id),
                no_webpage=True,
                entities=entities,
            )
        )
        return self._message_id_from_updates(result)

    async def publish(self, batch_id: str, body: str) -> int:
        async with self._publish_lock:
            async with self._client() as client:
                peer = await client.get_input_entity(self.settings.destination)
                return await self._send(client, peer, batch_id, body)

    async def probe_custom_emojis(self) -> ProbeResult:
        try:
            async with self._client() as client:
                peer = await client.get_input_entity("me")
                message_id = await self._send(
                    client,
                    peer,
                    "finko-custom-emoji-probe-v1",
                    "FINKO custom emoji probe",
                )
                message = await client.get_messages("me", ids=message_id)
                from telethon.tl import types

                ids = tuple(
                    int(entity.document_id)
                    for entity in (message.entities or [])
                    if isinstance(entity, types.MessageEntityCustomEmoji)
                )
                return ProbeResult(
                    ok=ids == EXPECTED_CUSTOM_EMOJI_IDS,
                    message_id=message_id,
                    returned_custom_emoji_ids=ids,
                    error=None if ids == EXPECTED_CUSTOM_EMOJI_IDS else "custom_emoji_entity_mismatch",
                )
        except Exception as exc:
            return ProbeResult(
                ok=False,
                message_id=None,
                returned_custom_emoji_ids=(),
                error=type(exc).__name__,
            )
