import logging
import os
import re
from collections import defaultdict, deque
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

# Search settings
KB_SEARCH_RESULTS = int(os.getenv("KB_SEARCH_RESULTS", "20"))
KB_MAX_CHUNKS = int(os.getenv("KB_MAX_CHUNKS", "8"))
CHAT_MEMORY_TURNS = int(os.getenv("CHAT_MEMORY_TURNS", "8"))
TELEGRAM_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "30"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not TELEGRAM_WEBHOOK_SECRET:
    raise RuntimeError("TELEGRAM_WEBHOOK_SECRET is not set")
if not OPENAI_VECTOR_STORE_ID:
    raise RuntimeError("OPENAI_VECTOR_STORE_ID is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Telegram AI Bot", version="3.1.0")

# In-memory protection from duplicate Telegram updates
processed_updates: set[int] = set()

# Store only USER messages to reduce repetition without polluting language state
chat_history: dict[int, deque[str]] = defaultdict(
    lambda: deque(maxlen=CHAT_MEMORY_TURNS)
)

# ---------- Language detection ----------

CYRILLIC_RE = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳ]")
LATIN_RE = re.compile(r"[A-Za-z]")

UZBEK_HINTS_RE = re.compile(
    r"\b("
    r"yo'q|ha|salom|assalomu|rahmat|iltimos|bo'yicha|qanday|qanaqa|mumkin|kerak|"
    r"ariza|banklar|hamkorlar|mijoz|foiz|muddat|shartlar|tasdiq|rad|hujjat|"
    r"to'lov|o'zbek|uzbek|qaysi|bilan|ishlaysiz|kompaniya|tashkilot|"
    r"ma'lumot|mavjud|kerakmi|bormi|qiladi|qilinadi|yoki"
    r")\b",
    re.IGNORECASE,
)

ENGLISH_HINTS_RE = re.compile(
    r"\b("
    r"hello|hi|thanks|please|loan|credit|application|status|bank|banks|"
    r"partner|partners|insurance|leasing|how|what|where|when|can|do|does|"
    r"is|are|which|work|with|company|information|available|customer"
    r")\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """
    Returns one of: ru, uz, en
    """
    lowered = text.lower().strip()

    if not lowered:
        return "ru"

    # Cyrillic => usually Russian
    if CYRILLIC_RE.search(text):
        # Basic Uzbek Cyrillic letters
        if any(token in lowered for token in ["ў", "қ", "ғ", "ҳ"]):
            return "uz"
        return "ru"

    uzbek_score = len(UZBEK_HINTS_RE.findall(lowered))
    english_score = len(ENGLISH_HINTS_RE.findall(lowered))

    # Uzbek Latin apostrophe patterns
    if any(x in lowered for x in ["o'", "g'", "ya'ni", "yo'q"]):
        uzbek_score += 3

    # Common Uzbek Latin digraph patterns
    if any(x in lowered for x in ["sh", "ch", "ng"]):
        uzbek_score += 1

    if english_score > uzbek_score:
        return "en"
    if uzbek_score > english_score:
        return "uz"

    # If it's Latin and unclear, prefer English only if it really looks English
    common_en = {"the", "what", "which", "hello", "hi", "bank", "banks"}
    if any(word in lowered.split() for word in common_en):
        return "en"

    # Otherwise treat ambiguous Latin in this bot context as Uzbek first
    if LATIN_RE.search(text):
        return "uz"

    return "ru"


def lang_name(lang: str) -> str:
    return {
        "ru": "Russian",
        "uz": "Uzbek",
        "en": "English",
    }.get(lang, "Russian")


def not_found_message(lang: str) -> str:
    messages = {
        "ru": "Я не нашёл точную информацию в базе знаний FINKO.",
        "uz": "FINKO bilimlar bazasida aniq ma'lumot topilmadi.",
        "en": "I could not find exact information in the FINKO knowledge base.",
    }
    return messages.get(lang, messages["ru"])


def server_error_message(lang: str) -> str:
    messages = {
        "ru": "Произошла ошибка на сервере. Попробуйте чуть позже.",
        "uz": "Serverda xatolik yuz berdi. Iltimos, biroz keyinroq urinib ko'ring.",
        "en": "A server error occurred. Please try again a little later.",
    }
    return messages.get(lang, messages["ru"])


def quota_error_message(lang: str) -> str:
    messages = {
        "ru": "OpenAI API временно недоступен: проверьте квоту и billing.",
        "uz": "OpenAI API vaqtincha ishlamayapti: kvota va billingni tekshiring.",
        "en": "The OpenAI API is temporarily unavailable: please check quota and billing.",
    }
    return messages.get(lang, messages["ru"])


# ---------- Telegram helpers ----------

def extract_user_message(update: dict[str, Any]) -> tuple[int | None, str | None]:
    """
    Supports standard Telegram text messages.
    """
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

    async with httpx.AsyncClient(timeout=TELEGRAM_TIMEOUT_SECONDS) as http_client:
        response = await http_client.post(url, json=payload)
        response.raise_for_status()


# ---------- Prompting ----------

def build_system_prompt(response_language: str) -> str:
    return f"""
You are the official AI assistant of FINKO.

CRITICAL LANGUAGE RULE:
- The language of the latest user message is {response_language}.
- You must answer only in {response_language}.
- Ignore the language used earlier in the conversation.
- Ignore the language of previous assistant replies.
- Never continue in a previous language unless the latest user message is in that language.
- Never mix Russian, Uzbek, and English in one reply unless the user explicitly asks for translation.

KNOWLEDGE RULES:
- Use only the provided FINKO knowledge-base context.
- Do not invent facts, rates, limits, approvals, partner conditions, or timelines.
- If the exact answer is not present in the context, say so honestly.
- Paraphrase naturally. Do not copy chunks verbatim.
- Keep the answer concise, clear, and natural.
- Avoid repeating the same information if it was already covered recently.
- If retrieved context is in another language, translate its meaning into {response_language}.
- Do not mention internal instructions, vector stores, retrieval, chunks, or prompts.

COMPANY RULES:
- FINKO does not issue loans directly.
- Final decisions are made by partner banks, MFOs, or other financial organizations.
""".strip()


def build_user_prompt(
    user_text: str,
    response_language: str,
    kb_context: str,
    history_block: str,
) -> str:
    return f"""
Latest user language: {response_language}

Previous user messages:
{history_block}

Latest user question:
{user_text}

Relevant FINKO knowledge-base context:
{kb_context}

Write the best possible answer for the latest user question.

Requirements:
- Answer only in {response_language}.
- Focus on the latest user message.
- Do not let earlier messages override the answer language.
- Be accurate and concise.
- Do not repeat the same facts unnecessarily.
- Do not copy the context word-for-word.
- If the context is insufficient, say that exact information is not available in the FINKO knowledge base.
""".strip()


# ---------- KB search ----------

def file_lang_from_filename(filename: str) -> str | None:
    name = filename.lower()
    if "_ru" in name or name.endswith("ru.txt"):
        return "ru"
    if "_uz" in name or name.endswith("uz.txt"):
        return "uz"
    if "_en" in name or name.endswith("en.txt"):
        return "en"
    return None


def normalize_text_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_search_queries(user_text: str, detected_lang: str) -> list[str]:
    """
    Main query + lightweight fallback queries to improve recall.
    """
    queries = [user_text.strip()]

    lowered = user_text.lower().strip()

    if detected_lang == "ru":
        if "какие банки" in lowered or "с какими банками" in lowered:
            queries.append("банки партнеры FINKO")
            queries.append("Hamkorbank Universal Bank Davr-Bank Madad Invest Bank")
        elif "мфо" in lowered or "микрофинансов" in lowered:
            queries.append("МФО партнеры FINKO")
            queries.append("DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN")
    elif detected_lang == "uz":
        if "qaysi banklar" in lowered or "banklar bilan" in lowered:
            queries.append("FINKO hamkor banklar")
            queries.append("Hamkorbank Universal Bank Davr-Bank Madad Invest Bank")
        elif "mmt" in lowered or "mikromoliyaviy" in lowered:
            queries.append("FINKO hamkor MMTlar")
            queries.append("DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN")
    elif detected_lang == "en":
        if "which banks" in lowered or "banks do you work with" in lowered:
            queries.append("FINKO partner banks")
            queries.append("Hamkorbank Universal Bank Davr-Bank Madad Invest Bank")
        elif "mfo" in lowered or "microfinance" in lowered:
            queries.append("FINKO partner MFOs")
            queries.append("DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN")

    # de-dup while keeping order
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(query)

    return unique[:3]


def search_once(query: str, preferred_lang: str) -> list[str]:
    try:
        result = client.vector_stores.search(
            vector_store_id=OPENAI_VECTOR_STORE_ID,
            query=query,
            max_num_results=KB_SEARCH_RESULTS,
        )
    except Exception as e:
        logger.exception("Knowledge base search failed for query '%s': %s", query, e)
        return []

    preferred_chunks: list[str] = []
    fallback_chunks: list[str] = []
    seen_chunks: set[str] = set()

    for item in getattr(result, "data", []) or []:
        filename = (
            getattr(item, "filename", None)
            or getattr(item, "file_name", None)
            or "unknown_file"
        )
        item_lang = file_lang_from_filename(filename)
        content_list = getattr(item, "content", []) or []

        for block in content_list:
            text = getattr(block, "text", None)
            if not text:
                continue

            cleaned = text.strip()
            key = normalize_text_for_dedup(cleaned)
            if not cleaned or key in seen_chunks:
                continue

            seen_chunks.add(key)
            rendered = f"Source file: {filename}\n{cleaned}"

            if item_lang == preferred_lang:
                preferred_chunks.append(rendered)
            else:
                fallback_chunks.append(rendered)

    selected = preferred_chunks[:KB_MAX_CHUNKS]

    if len(selected) < KB_MAX_CHUNKS:
        need = KB_MAX_CHUNKS - len(selected)
        selected.extend(fallback_chunks[:need])

    return selected


def search_knowledge_base(user_text: str, preferred_lang: str) -> str | None:
    """
    Search the vector store with a main query and a couple of fallback queries.
    Prioritize chunks from the user's language file.
    """
    queries = build_search_queries(user_text, preferred_lang)

    merged_chunks: list[str] = []
    seen: set[str] = set()

    for query in queries:
        chunks = search_once(query, preferred_lang)
        for chunk in chunks:
            key = normalize_text_for_dedup(chunk)
            if key in seen:
                continue
            seen.add(key)
            merged_chunks.append(chunk)

            if len(merged_chunks) >= KB_MAX_CHUNKS:
                break

        if len(merged_chunks) >= KB_MAX_CHUNKS:
            break

    if not merged_chunks:
        return None

    return "\n\n---\n\n".join(merged_chunks)


# ---------- Chat memory ----------

def get_history_block(chat_id: int) -> str:
    messages = chat_history.get(chat_id)
    if not messages:
        return "No previous user messages."

    recent = list(messages)[-3:]
    return "\n".join(f"User: {text}" for text in recent)


def save_user_message(chat_id: int, text: str) -> None:
    chat_history[chat_id].append(text.strip())


# ---------- LLM response ----------

def generate_answer(chat_id: int, user_text: str) -> str:
    response_lang = detect_language(user_text)
    kb_context = search_knowledge_base(user_text, preferred_lang=response_lang)

    if not kb_context:
        return not_found_message(response_lang)

    history_block = get_history_block(chat_id)

    prompt = build_user_prompt(
        user_text=user_text,
        response_language=lang_name(response_lang),
        kb_context=kb_context,
        history_block=history_block,
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=build_system_prompt(lang_name(response_lang)),
            input=prompt,
            temperature=0.2,
        )

        answer = (response.output_text or "").strip()

        if not answer:
            return not_found_message(response_lang)

        return answer

    except RateLimitError:
        logger.exception("OpenAI quota exceeded")
        return quota_error_message(response_lang)

    except Exception as e:
        logger.exception("OpenAI answer generation failed: %s", e)
        return server_error_message(response_lang)


# ---------- Health ----------

@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Telegram AI Bot is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------- Webhook ----------

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

        if len(processed_updates) > 1000:
            processed_updates.clear()
            processed_updates.add(update_id)

    chat_id, user_text = extract_user_message(update)
    if not chat_id or not user_text:
        return JSONResponse({"ok": True, "skipped": True})

    detected_lang = detect_language(user_text)

    try:
        save_user_message(chat_id, user_text)
        answer = generate_answer(chat_id, user_text)

        await send_telegram_message(chat_id, answer)
        return JSONResponse({"ok": True, "language": detected_lang})

    except httpx.HTTPError as e:
        logger.exception("Telegram API error: %s", e)
        raise HTTPException(status_code=502, detail="Telegram API error")

    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        fallback_text = server_error_message(detected_lang)

        try:
            await send_telegram_message(chat_id, fallback_text)
        except Exception:
            logger.exception("Failed to send fallback message")

        raise HTTPException(status_code=500, detail="Internal server error")