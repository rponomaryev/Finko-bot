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
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
KB_MAX_CHUNKS = int(os.getenv("KB_MAX_CHUNKS", "5"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not TELEGRAM_WEBHOOK_SECRET:
    raise RuntimeError("TELEGRAM_WEBHOOK_SECRET is not set")
if not OPENAI_VECTOR_STORE_ID:
    raise RuntimeError("OPENAI_VECTOR_STORE_ID is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Telegram AI Bot", version="2.0.0")

# Защита от дублей update_id
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
        "Ты — AI-ассистент компании Finko. "
        "Отвечай только на основе найденного контекста из базы знаний Finko. "
        "Не выдумывай факты, суммы, сроки, адреса, статусы, проценты и обещания. "
        "Если точного ответа в базе знаний нет, честно скажи: "
        "'Я не нашёл точную информацию в базе знаний Finko.' "
        "Отвечай простым, понятным и дружелюбным русским языком. "
        "Если пользователь спрашивает про услуги, объясняй коротко и по делу."
    )


def search_knowledge_base(user_text: str) -> str | None:
    """
    Ищет релевантные куски текста в OpenAI Vector Store.
    """
    try:
        result = client.vector_stores.search(
            vector_store_id=OPENAI_VECTOR_STORE_ID,
            query=user_text,
            max_num_results=KB_MAX_CHUNKS,
        )
    except Exception as e:
        logger.exception("Knowledge base search failed: %s", e)
        return None

    chunks: list[str] = []

    for item in getattr(result, "data", []) or []:
        filename = getattr(item, "filename", None) or getattr(item, "file_name", None) or "unknown_file"
        content_list = getattr(item, "content", []) or []

        for block in content_list:
            text = getattr(block, "text", None)
            if text:
                chunks.append(f"Источник: {filename}\n{text.strip()}")

    if not chunks:
        return None

    return "\n\n---\n\n".join(chunks[:KB_MAX_CHUNKS])


def generate_answer(user_text: str) -> str:
    kb_context = search_knowledge_base(user_text)

    if not kb_context:
        return "Я не нашёл точную информацию в базе знаний Finko."

    prompt = f"""
Вопрос пользователя:
{user_text}

Контекст из базы знаний Finko:
{kb_context}

Сформируй ответ только по этому контексту.
Если контекст неполный или точного ответа нет, честно скажи:
"Я не нашёл точную информацию в базе знаний Finko."
""".strip()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=build_system_prompt(),
            input=prompt,
            temperature=0.2,
        )

        answer = (response.output_text or "").strip()

        if not answer:
            return "Не получилось сформировать ответ. Попробуйте ещё раз."

        return answer

    except RateLimitError:
        logger.exception("OpenAI quota exceeded")
        return "OpenAI API сейчас недоступен: закончилась квота или не настроен billing."

    except Exception as e:
        logger.exception("OpenAI answer generation failed: %s", e)
        return "Произошла ошибка при формировании ответа. Попробуйте чуть позже."


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
    logger.info("Incoming update received")

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
        answer = generate_answer(user_text)
        await send_telegram_message(chat_id, answer)
        return JSONResponse({"ok": True})

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