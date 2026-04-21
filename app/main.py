import logging
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from openai import OpenAI, RateLimitError

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("telegram-ai-bot")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not TELEGRAM_WEBHOOK_SECRET:
    raise RuntimeError("TELEGRAM_WEBHOOK_SECRET is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Telegram AI Bot", version="1.0.0")

# Храним уже обработанные update_id, чтобы бот не отвечал по несколько раз
processed_updates: set[int] = set()


def extract_user_message(update: dict[str, Any]) -> tuple[int | None, str | None]:
    message = update.get("message")
    if not message:
        return None, None

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text")

    if not chat_id or not text:
        return None, None

    return chat_id, text.strip()


async def send_telegram_message(chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
    }

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        response = await http_client.post(url, json=payload)
        response.raise_for_status()


def build_system_prompt() -> str:
    return (
        "Ты — вежливый AI-ассистент компании. "
        "Отвечай кратко, понятно и по делу. "
        "Не здоровайся в каждом сообщении заново, если пользователь уже начал диалог. "
        "Если не уверен, так и скажи. "
        "Не выдумывай факты о компании."
    )


def ask_openai(user_text: str) -> str:
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=build_system_prompt(),
        input=user_text,
        temperature=0.3,
    )

    answer = response.output_text
    if not answer:
        return "Не получилось сформировать ответ. Попробуйте ещё раз."
    return answer.strip()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Telegram AI Bot is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Telegram secret token",
        )

    update = await request.json()
    logger.info("Incoming update: %s", update)

    update_id = update.get("update_id")
    if update_id is not None:
        if update_id in processed_updates:
            logger.info("Duplicate update skipped: %s", update_id)
            return JSONResponse({"ok": True, "duplicate": True})

        processed_updates.add(update_id)

        # чтобы set не рос бесконечно
        if len(processed_updates) > 1000:
            processed_updates.clear()
            processed_updates.add(update_id)

    chat_id, user_text = extract_user_message(update)
    if not chat_id or not user_text:
        return JSONResponse({"ok": True, "skipped": True})

    try:
        answer = ask_openai(user_text)
        await send_telegram_message(chat_id, answer)
        return JSONResponse({"ok": True})

    except RateLimitError:
        logger.exception("OpenAI quota exceeded")
        await send_telegram_message(
            chat_id,
            "OpenAI API сейчас недоступен: закончилась квота или не настроен billing.",
        )
        return JSONResponse({"ok": True, "error": "openai_quota_exceeded"})

    except httpx.HTTPError as e:
        logger.exception("Telegram API error: %s", e)
        raise HTTPException(status_code=502, detail="Telegram API error")

    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        try:
            await send_telegram_message(
                chat_id,
                "Произошла ошибка на сервере. Попробуйте чуть позже.",
            )
        except Exception:
            logger.exception("Failed to send fallback message")
        raise HTTPException(status_code=500, detail="Internal server error")