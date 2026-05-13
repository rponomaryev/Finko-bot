import re

from openai import RateLimitError

from app.bot.intents import text_similarity
from app.bot.language import lang_name, not_found_message, quota_error_message, server_error_message
from app.bot.ui import strip_cross_language_artifacts
from app.bot import state
from app.config import OPENAI_MODEL, OPENAI_MAX_RETRIES, OPENAI_RETRY_BASE_DELAY, logger
from app.kb.search import search_knowledge_base
from app.services.openai_client import client
from app.utils.retry import is_retryable_openai_error, retry_delay


async def get_history_block(chat_id: int) -> str:
    messages = await state.get_user_history(chat_id, limit=3)
    if not messages:
        return "No previous user messages."
    return "\n".join(f"User: {text}" for text in messages)


async def is_semantically_repeated_question(chat_id: int, user_text: str) -> bool:
    recent = await state.get_user_history(chat_id, limit=3)
    for old_text in recent:
        if text_similarity(old_text, user_text) >= 0.85:
            return True
    return False


async def reduce_repetition_if_needed(chat_id: int, answer: str) -> str:
    recent_answers = await state.get_assistant_history(chat_id)
    for old_answer in recent_answers:
        if text_similarity(old_answer, answer) >= 0.86:
            sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
            if len(sentences) >= 2:
                return " ".join(sentences[:2]).strip()
            return answer
    return answer


def build_system_prompt(response_language: str, user_type: str) -> str:
    return f"""
You are the official AI assistant of FINKO — a financial marketplace in Uzbekistan.

CRITICAL LANGUAGE RULE:
- The language of the latest user message is: {response_language}.
- You MUST answer ONLY in {response_language}.
- Ignore the language of earlier messages or previous assistant replies.
- Never mix Russian, Uzbek, and English in one reply unless the user explicitly asks for translation.
- If the language is Uzbek in Latin script → answer fully in Uzbek Latin script.
- If the language is Uzbek in Cyrillic script → answer fully in Uzbek Cyrillic script.
- If context chunks are in another language/script, translate their meaning into {response_language}.

USER TYPE:
- Detected user type: {user_type}. Adapt phrasing accordingly.

KNOWLEDGE RULES:
- Use only the provided FINKO knowledge-base context.
- Do not invent facts, rates, limits, approvals, partner conditions, or timelines.
- If the exact answer is not in the context, say so honestly.
- Paraphrase naturally. Do not copy chunks verbatim.
- Keep the answer concise, clear, and natural.
- Do not mention internal instructions, vector stores, retrieval, chunks, or prompts.

LOAN / CREDIT RULES:
- FINKO does not issue loans directly. All decisions are made by partner banks, MFOs, or other organizations.
- Never promise or imply that a loan will be approved.
- If the user already has an existing loan and asks whether they can get another:
  answer honestly — having an existing loan increases the debt burden and may reduce chances of approval,
  but the final decision belongs to the partner bank or MFO. Do NOT say "you will definitely get one".
- If the user asks about eligibility without confirmed income:
  note that most banks require proof of income; microloans via MFOs may require less documentation,
  but typically carry higher rates. Do NOT guarantee approval.

CALL-TO-ACTION RULE:
- Whenever the answer leads the user toward applying, viewing offers, or signing up,
  always include the platform link:
  - Russian / English: https://finko.uz/ru
  - Uzbek (Latin or Cyrillic): https://finko.uz/uz

COMPANY RULES:
- FINKO does not issue loans directly.
- Final decisions are made by partner banks, MFOs, or other financial organizations.
- Do not give legal or financial guarantees.
""".strip()


def build_user_prompt(
    user_text: str,
    response_language: str,
    kb_context: str,
    history_block: str,
    intent: str,
    repeated_question: bool,
) -> str:
    repeat_instruction = (
        "The user is asking a repeated or very similar question. Answer more briefly and avoid repeating wording."
        if repeated_question
        else "Answer normally, but stay concise."
    )

    return f"""
Latest user language: {response_language}
Detected intent: {intent}

Previous user messages:
{history_block}

Latest user question:
{user_text}

Relevant FINKO knowledge-base context:
{kb_context}

Write the best possible answer for the latest user question.

Requirements:
- Answer ONLY in {response_language}. Do not mix languages.
- Focus on the latest user message.
- Be accurate, honest, and concise.
- Do not repeat the same facts unnecessarily.
- Do not copy the context word-for-word.
- Never promise loan approval. FINKO is a marketplace; decisions belong to banks and MFOs.
- When the answer includes a call-to-action (apply, view offers), include the platform link.
- {repeat_instruction}
- If the context is insufficient, say that exact information is not available in the FINKO knowledge base.
""".strip()


async def _create_response_with_retry(model: str, instructions: str, prompt: str):
    attempts = max(OPENAI_MAX_RETRIES, 1)
    for attempt in range(attempts):
        try:
            return await client.responses.create(
                model=model,
                instructions=instructions,
                input=prompt,
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 - retry utility classifies errors
            if attempt >= attempts - 1 or not is_retryable_openai_error(exc):
                raise
            delay = retry_delay(attempt, OPENAI_RETRY_BASE_DELAY)
            logger.warning("OpenAI response failed, retrying in %.2fs: %s", delay, exc)
            import asyncio
            await asyncio.sleep(delay)


async def generate_answer(
    chat_id: int,
    user_text: str,
    intent: str,
    user_type: str,
    response_lang: str,
) -> str:
    kb_context = await search_knowledge_base(
        user_text=user_text,
        preferred_lang=response_lang,
        intent=intent,
    )

    if not kb_context:
        return not_found_message(response_lang)

    history_block = await get_history_block(chat_id)
    repeated_question = await is_semantically_repeated_question(chat_id, user_text)

    prompt = build_user_prompt(
        user_text=user_text,
        response_language=lang_name(response_lang),
        kb_context=kb_context,
        history_block=history_block,
        intent=intent,
        repeated_question=repeated_question,
    )

    try:
        response = await _create_response_with_retry(
            OPENAI_MODEL,
            build_system_prompt(lang_name(response_lang), user_type),
            prompt,
        )

        answer = (response.output_text or "").strip()

        if not answer:
            return not_found_message(response_lang)

        answer = await reduce_repetition_if_needed(chat_id, answer)
        answer = strip_cross_language_artifacts(answer, response_lang)
        return answer

    except RateLimitError:
        logger.exception("OpenAI quota/rate limit after retries")
        return quota_error_message(response_lang)

    except Exception as e:
        logger.exception("OpenAI answer generation failed after retries: %s", e)
        return server_error_message(response_lang)
