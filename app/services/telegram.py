import asyncio
from typing import Any

import httpx

from app.bot.ui import get_keyboard_for_lang
from app.config import (
    MINI_APP_BUTTON_TEXT,
    MINI_APP_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_MAX_RETRIES,
    TELEGRAM_RETRY_BASE_DELAY,
    TELEGRAM_TIMEOUT_SECONDS,
    logger,
)
from app.utils.retry import retry_delay

telegram_client: httpx.AsyncClient | None = None


async def init_telegram_client() -> None:
    global telegram_client
    telegram_client = httpx.AsyncClient(timeout=TELEGRAM_TIMEOUT_SECONDS)


async def close_telegram_client() -> None:
    global telegram_client
    if telegram_client is not None:
        await telegram_client.aclose()
        telegram_client = None


def get_telegram_client() -> httpx.AsyncClient:
    if telegram_client is None:
        raise RuntimeError("Telegram HTTP client is not initialized")
    return telegram_client


def _telegram_retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _is_retryable_telegram_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


async def _post_telegram(url: str, payload: dict[str, Any]) -> httpx.Response:
    attempts = max(TELEGRAM_MAX_RETRIES, 1)
    http_client = get_telegram_client()
    for attempt in range(attempts):
        try:
            response = await http_client.post(url, json=payload)
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - retry classifier handles details
            if attempt >= attempts - 1 or not _is_retryable_telegram_error(exc):
                raise
            retry_after = _telegram_retry_after(exc.response) if isinstance(exc, httpx.HTTPStatusError) else None
            delay = retry_delay(attempt, TELEGRAM_RETRY_BASE_DELAY, retry_after=retry_after)
            logger.warning("Telegram API call failed, retrying in %.2fs: %s", delay, exc)
            await asyncio.sleep(delay)
    raise RuntimeError("Telegram request failed without exception")


async def set_telegram_menu_button() -> None:
    if not MINI_APP_URL:
        logger.warning("MINI_APP_URL is not set, Telegram menu button skipped")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setChatMenuButton"
    payload = {
        "menu_button": {
            "type": "web_app",
            "text": MINI_APP_BUTTON_TEXT,
            "web_app": {"url": MINI_APP_URL},
        }
    }

    await _post_telegram(url, payload)
    logger.info("Telegram menu button configured: %s -> %s", MINI_APP_BUTTON_TEXT, MINI_APP_URL)


async def send_telegram_message(
    chat_id: int,
    text: str,
    ui_lang: str = "ru",
    custom_keyboard: dict[str, Any] | None = None,
) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": custom_keyboard if custom_keyboard is not None else get_keyboard_for_lang(ui_lang),
    }
    await _post_telegram(url, payload)
