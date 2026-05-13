from typing import Any

import json

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.bot import state
from app.bot.calculator import handle_loan_calc
from app.bot.intents import detect_intent, infer_user_type
from app.bot.language import detect_language, rate_limit_message, server_error_message, too_long_message
from app.bot.quick_answers import handle_menu_or_quick_action
from app.bot.telegram_update import extract_user_message
from app.bot.ui import (
    build_language_clarification_text,
    build_language_saved_text,
    build_start_language_text,
    get_language_keyboard,
    handle_language_selection,
)
from app.config import MAX_MESSAGE_CHARS, MAX_WEBHOOK_BODY_BYTES, TELEGRAM_WEBHOOK_SECRET, logger
from app.db.repositories import get_user_ui_language, log_event, log_message, upsert_user_profile
from app.services.openai_service import generate_answer
from app.services.telegram import send_telegram_message

router = APIRouter()

def validate_webhook_body_size(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if not content_length:
        return

    try:
        size = int(content_length)
    except ValueError:
        return

    if MAX_WEBHOOK_BODY_BYTES > 0 and size > MAX_WEBHOOK_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Webhook body is too large",
        )


async def read_limited_json_body(request: Request) -> dict[str, Any]:
    body = await request.body()

    if MAX_WEBHOOK_BODY_BYTES > 0 and len(body) > MAX_WEBHOOK_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Webhook body is too large",
        )

    try:
        parsed = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Telegram update body",
        )

    return parsed


@router.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Telegram secret token")

    validate_webhook_body_size(request)
    update = await read_limited_json_body(request)
    logger.info("Incoming update received")

    update_id = update.get("update_id")
    if update_id is not None:
        if not await state.remember_update_id(int(update_id)):
            logger.info("Duplicate update skipped: %s", update_id)
            return JSONResponse({"ok": True, "duplicate": True})

    chat_id, user_text, profile = extract_user_message(update)
    if not chat_id or not user_text:
        return JSONResponse({"ok": True, "skipped": True})

    selected_ui_lang = await get_user_ui_language(chat_id)
    fallback_lang = selected_ui_lang or detect_language(user_text)
    if fallback_lang == "unknown":
        fallback_lang = "ru"

    if len(user_text) > MAX_MESSAGE_CHARS:
        if await state.can_send_rate_limit_notice(chat_id):
            await send_telegram_message(chat_id, too_long_message(fallback_lang), ui_lang=fallback_lang)
        await log_event(chat_id, "message_too_long", str(len(user_text)))
        return JSONResponse({"ok": True, "source": "message_too_long"})

    if await state.is_chat_rate_limited(chat_id):
        if await state.can_send_rate_limit_notice(chat_id):
            await send_telegram_message(chat_id, rate_limit_message(fallback_lang), ui_lang=fallback_lang)
        await log_event(chat_id, "chat_rate_limited", user_text[:120])
        return JSONResponse({"ok": True, "source": "chat_rate_limited"})

    if user_text == "/start":
        await send_telegram_message(
            chat_id, build_start_language_text(),
            ui_lang="ru", custom_keyboard=get_language_keyboard(),
        )
        await log_event(chat_id, "start_shown", "language_selector")
        return JSONResponse({"ok": True, "source": "start_language_selector"})

    selected_lang = handle_language_selection(user_text)
    if selected_lang:
        await upsert_user_profile(
            chat_id=chat_id,
            username=profile.get("username"),
            first_name=profile.get("first_name"),
            last_name=profile.get("last_name"),
            language=selected_lang,
            user_type="customer",
            selected_language=selected_lang,
        )
        await send_telegram_message(chat_id, build_language_saved_text(selected_lang), ui_lang=selected_lang)
        await log_event(chat_id, "ui_language_selected", selected_lang)
        return JSONResponse({"ok": True, "selected_language": selected_lang})

    message_lang = detect_language(user_text)

    if message_lang == "unknown":
        fallback_ui_lang = selected_ui_lang or "ru"
        await send_telegram_message(
            chat_id, build_language_clarification_text(),
            ui_lang=fallback_ui_lang, custom_keyboard=get_language_keyboard(),
        )
        await log_event(chat_id, "language_clarification_requested", user_text)
        return JSONResponse({"ok": True, "source": "language_clarification"})

    ui_lang = selected_ui_lang or message_lang
    response_lang = message_lang
    intent = detect_intent(user_text)
    user_type = infer_user_type(intent, user_text)

    await upsert_user_profile(
        chat_id=chat_id,
        username=profile.get("username"),
        first_name=profile.get("first_name"),
        last_name=profile.get("last_name"),
        language=message_lang,
        user_type=user_type,
        selected_language=None,
    )

    await log_message(
        chat_id=chat_id,
        direction="inbound",
        text=user_text,
        language=message_lang,
        intent=intent,
        user_type=user_type,
        source="telegram",
    )

    # 1. Quick menu replies (greeting, contacts, etc.)
    quick_answer, quick_intent = await handle_menu_or_quick_action(user_text, chat_id, response_lang)
    if quick_answer:
        await send_telegram_message(chat_id, quick_answer, ui_lang=ui_lang)
        await state.save_assistant_message(chat_id, quick_answer)
        await log_message(chat_id=chat_id, direction="outbound", text=quick_answer,
                    language=response_lang, intent=quick_intent, user_type=user_type, source="quick")
        await log_event(chat_id, "quick_reply", quick_intent)
        return JSONResponse({"ok": True, "source": "quick", "intent": quick_intent})

    # 2. Local loan calculator (no OpenAI call needed)
    calc_answer = await handle_loan_calc(chat_id, user_text, response_lang)
    if calc_answer:
        await send_telegram_message(chat_id, calc_answer, ui_lang=ui_lang)
        await state.save_assistant_message(chat_id, calc_answer)
        await log_message(chat_id=chat_id, direction="outbound", text=calc_answer,
                    language=response_lang, intent="loan_calc", user_type=user_type, source="calculator")
        await log_event(chat_id, "loan_calc_reply", intent)
        return JSONResponse({"ok": True, "source": "calculator", "intent": "loan_calc"})

    # 3. OpenAI + knowledge base
    openai_allowed, openai_limit_reason = await state.reserve_openai_call(chat_id)
    if not openai_allowed:
        if await state.can_send_rate_limit_notice(chat_id):
            await send_telegram_message(chat_id, rate_limit_message(ui_lang, openai_limit_reason or "openai_limited"), ui_lang=ui_lang)
        await log_event(chat_id, "openai_rate_limited", openai_limit_reason)
        return JSONResponse({"ok": True, "source": "openai_rate_limited", "reason": openai_limit_reason})

    try:
        await state.save_user_message(chat_id, user_text)

        answer = await generate_answer(
            chat_id=chat_id,
            user_text=user_text,
            intent=intent,
            user_type=user_type,
            response_lang=response_lang,
        )

        await send_telegram_message(chat_id, answer, ui_lang=ui_lang)
        await state.save_assistant_message(chat_id, answer)

        await log_message(chat_id=chat_id, direction="outbound", text=answer,
                    language=message_lang, intent=intent, user_type=user_type, source="openai")
        await log_event(chat_id, "openai_reply", intent)

        return JSONResponse({
            "ok": True,
            "message_language": message_lang,
            "response_language": response_lang,
            "ui_language": ui_lang,
            "intent": intent,
            "user_type": user_type,
            "source": "openai",
        })

    except httpx.HTTPError as e:
        logger.exception("Telegram API error: %s", e)
        raise HTTPException(status_code=502, detail="Telegram API error")

    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        fallback_text = server_error_message(message_lang)
        try:
            await send_telegram_message(chat_id, fallback_text, ui_lang=ui_lang)
            await log_message(chat_id=chat_id, direction="outbound", text=fallback_text,
                        language=message_lang, intent=intent, user_type=user_type, source="fallback")
            await log_event(chat_id, "fallback_reply", str(e))
        except Exception:
            logger.exception("Failed to send fallback message")
        raise HTTPException(status_code=500, detail="Internal server error")
