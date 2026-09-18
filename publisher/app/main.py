import hmac
import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .telegram_client import TelegramPublisher


logger = logging.getLogger("finko.publisher")
settings = Settings.from_env()
PUBLISH_KEY = settings.publish_key
publisher = TelegramPublisher(settings)
app = FastAPI(title="FINKO Telegram Publisher")


class PublishRequest(BaseModel):
    batch_id: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=3600)


def _authorized(value: str | None) -> bool:
    return bool(value) and hmac.compare_digest(value, PUBLISH_KEY)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/preview")
async def preview(
    payload: PublishRequest,
    x_finko_publish_key: str | None = Header(default=None),
):
    if not _authorized(x_finko_publish_key):
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        message_id = await publisher.preview(payload.batch_id, payload.text)
    except Exception as exc:
        logger.exception(
            "preview_failed batch_id=%s error_type=%s",
            payload.batch_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="telegram_preview_failed") from None

    return {
        "ok": True,
        "batch_id": payload.batch_id,
        "chat": str(settings.review_destination),
        "message_id": message_id,
    }


@app.post("/publish")
async def publish(
    payload: PublishRequest,
    x_finko_publish_key: str | None = Header(default=None),
):
    if not _authorized(x_finko_publish_key):
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        message_id = await publisher.publish(payload.batch_id, payload.text)
    except Exception as exc:
        logger.exception(
            "publish_failed batch_id=%s error_type=%s",
            payload.batch_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="telegram_publish_failed") from None

    return {
        "ok": True,
        "batch_id": payload.batch_id,
        "chat": settings.destination,
        "message_id": message_id,
    }
