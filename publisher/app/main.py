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


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/publish")
async def publish(
    payload: PublishRequest,
    x_finko_publish_key: str | None = Header(default=None),
):
    if not x_finko_publish_key or not hmac.compare_digest(
        x_finko_publish_key,
        PUBLISH_KEY,
    ):
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

    logger.info(
        "published batch_id=%s message_id=%s",
        payload.batch_id,
        message_id,
    )
    return {
        "ok": True,
        "batch_id": payload.batch_id,
        "chat": "@finkouz",
        "message_id": message_id,
    }
