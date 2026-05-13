import re

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.config import (
    KB_MAX_CHUNKS,
    KB_SEARCH_RESULTS,
    OPENAI_MAX_RETRIES,
    OPENAI_RETRY_BASE_DELAY,
    OPENAI_VECTOR_STORE_ID,
    logger,
)
from app.services.openai_client import client
from app.utils.retry import retry_delay

def file_lang_from_filename(filename: str) -> str | None:
    name = filename.lower()

    if "_uz_cyrl" in name or "uz_cyrl" in name:
        return "uz_cyrl"
    if "_uz" in name or name.endswith("uz.txt"):
        return "uz_latn"
    if "_ru" in name or name.endswith("ru.txt"):
        return "ru"
    if "_en" in name or name.endswith("en.txt"):
        return "en"

    return None


def normalize_text_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def preferred_search_languages(input_lang: str) -> list[str]:
    if input_lang == "uz_cyrl":
        return ["uz_cyrl", "uz_latn", "ru", "en"]
    if input_lang == "uz_latn":
        return ["uz_latn", "uz_cyrl", "ru", "en"]
    if input_lang == "ru":
        return ["ru", "uz_cyrl", "uz_latn", "en"]
    return ["en", "ru", "uz_latn", "uz_cyrl"]


def build_search_queries(user_text: str, detected_lang: str, intent: str) -> list[str]:
    queries = [user_text.strip()]

    if intent in {"partners", "partners_menu"}:
        if detected_lang == "ru":
            queries.extend([
                "банки партнеры FINKO",
                "Hamkorbank Universal Bank DavrBank Madad Invest Bank Garant Bank Tenge Bank Asia Alliance Bank",
                "МФО партнеры FINKO DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN",
            ])
        elif detected_lang in {"uz_latn", "uz_cyrl"}:
            queries.extend([
                "FINKO hamkor banklar",
                "Hamkorbank Universal Bank DavrBank Madad Invest Bank Garant Bank Tenge Bank Asia Alliance Bank",
                "FINKO hamkor MMTlar DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN",
            ])
        else:
            queries.extend([
                "FINKO partner banks",
                "Hamkorbank Universal Bank DavrBank Madad Invest Bank Garant Bank Tenge Bank Asia Alliance Bank",
                "FINKO partner MFOs DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN",
            ])

    elif intent in {"business", "business_menu"}:
        if detected_lang == "ru":
            queries.extend(["бизнес кредиты FINKO", "лизинг страхование бизнес FINKO"])
        elif detected_lang in {"uz_latn", "uz_cyrl"}:
            queries.extend(["FINKO biznes kreditlari", "lizing sug'urta biznes FINKO"])
        else:
            queries.extend(["FINKO business loans", "FINKO leasing insurance business"])

    elif intent in {"credits", "credits_menu"}:
        if detected_lang == "ru":
            queries.extend([
                "кредиты FINKO",
                "как подать заявку на кредит FINKO",
                "ипотека автокредит микрозаймы FINKO",
            ])
        elif detected_lang in {"uz_latn", "uz_cyrl"}:
            queries.extend([
                "FINKO kreditlar",
                "kredit uchun ariza topshirish FINKO",
                "ipoteka avtokredit mikrozaym FINKO",
            ])
        else:
            queries.extend([
                "FINKO credits",
                "how to apply for a loan on FINKO",
            ])

    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(query)

    return unique[:8]


async def _vector_store_search_with_retry(query: str):
    attempts = max(OPENAI_MAX_RETRIES, 1)
    for attempt in range(attempts):
        try:
            result_items = []
            async for item in client.vector_stores.search(
                vector_store_id=OPENAI_VECTOR_STORE_ID,
                query=query,
                max_num_results=KB_SEARCH_RESULTS,
            ):
                result_items.append(item)
            return result_items
        except Exception as exc:  # noqa: BLE001 - retry utility classifies errors
            retryable = isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError))
            if isinstance(exc, APIStatusError):
                retryable = exc.status_code == 429 or exc.status_code >= 500
            if attempt >= attempts - 1 or not retryable:
                raise
            delay = retry_delay(attempt, OPENAI_RETRY_BASE_DELAY)
            logger.warning("KB search failed, retrying in %.2fs for query '%s': %s", delay, query, exc)
            import asyncio
            await asyncio.sleep(delay)


async def search_once(query: str, preferred_lang: str) -> list[str]:
    try:
        result_items = await _vector_store_search_with_retry(query)
    except Exception as e:
        logger.exception("Knowledge base search failed for query '%s' after retries: %s", query, e)
        return []

    language_priority = preferred_search_languages(preferred_lang)
    grouped_chunks: dict[str, list[str]] = {lang: [] for lang in language_priority}
    grouped_chunks["other"] = []

    seen_chunks: set[str] = set()

    for item in result_items:
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

            if item_lang in grouped_chunks:
                grouped_chunks[item_lang].append(rendered)
            else:
                grouped_chunks["other"].append(rendered)

    selected: list[str] = []

    for lang in language_priority:
        for chunk in grouped_chunks.get(lang, []):
            selected.append(chunk)
            if len(selected) >= KB_MAX_CHUNKS:
                return selected

    for chunk in grouped_chunks["other"]:
        selected.append(chunk)
        if len(selected) >= KB_MAX_CHUNKS:
            return selected

    return selected

async def search_knowledge_base(user_text: str, preferred_lang: str, intent: str) -> str | None:
    queries = build_search_queries(user_text, preferred_lang, intent)

    merged_chunks: list[str] = []
    seen: set[str] = set()

    for query in queries:
        chunks = await search_once(query, preferred_lang)
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
