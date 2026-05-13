import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Any

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - deployment installs redis from requirements.txt
    redis = None

from app.config import (
    CHAT_MEMORY_TURNS,
    OPENAI_COOLDOWN_SECONDS,
    OPENAI_GLOBAL_MAX_CALLS,
    OPENAI_GLOBAL_WINDOW_SECONDS,
    PROCESSED_UPDATES_MAXLEN,
    RATE_LIMIT_MAX_MESSAGES,
    RATE_LIMIT_NOTICE_COOLDOWN_SECONDS,
    RATE_LIMIT_WINDOW_SECONDS,
    REDIS_KEY_PREFIX,
    REDIS_URL,
    logger,
)

# All state APIs are async, even when using the in-memory fallback. This keeps
# handler code consistent and avoids mixing synchronous locks with asyncio code.
_state_lock = asyncio.Lock()
_redis = None

processed_updates: deque[int] = deque(maxlen=PROCESSED_UPDATES_MAXLEN)
processed_update_set: set[int] = set()

chat_user_history: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=CHAT_MEMORY_TURNS))
chat_assistant_history: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=CHAT_MEMORY_TURNS))
chat_loan_calc_pending: dict[int, dict[str, Any]] = {}

chat_message_windows: dict[int, deque[float]] = defaultdict(deque)
chat_last_openai_at: dict[int, float] = {}
chat_last_rate_limit_notice_at: dict[int, float] = {}
global_openai_window: deque[float] = deque()


def _key(*parts: object) -> str:
    return ":".join([REDIS_KEY_PREFIX, *[str(part) for part in parts]])


async def init_state_store() -> None:
    global _redis
    if not REDIS_URL:
        logger.warning("REDIS_URL is not set; using in-memory state fallback")
        return
    if redis is None:
        logger.warning("redis package is not installed; using in-memory state fallback")
        return
    try:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
        await _redis.ping()
        logger.info("Redis state store connected")
    except Exception as exc:  # noqa: BLE001 - fallback should be resilient
        logger.exception("Redis connection failed, falling back to in-memory state: %s", exc)
        _redis = None


async def close_state_store() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _prune_old_timestamps(window: deque[float], now: float, max_age_seconds: float) -> None:
    cutoff = now - max_age_seconds
    while window and window[0] < cutoff:
        window.popleft()


async def remember_update_id(update_id: int) -> bool:
    """Return False if this Telegram update was already processed."""
    if _redis is not None:
        key = _key("processed_updates")
        member = str(update_id)
        if await _redis.zscore(key, member) is not None:
            return False
        await _redis.zadd(key, {member: time.time()})
        await _redis.zremrangebyrank(key, 0, -(PROCESSED_UPDATES_MAXLEN + 1))
        await _redis.expire(key, 7 * 24 * 60 * 60)
        return True

    async with _state_lock:
        if update_id in processed_update_set:
            return False
        if len(processed_updates) == processed_updates.maxlen:
            processed_update_set.discard(processed_updates[0])
        processed_updates.append(update_id)
        processed_update_set.add(update_id)
        return True


async def save_user_message(chat_id: int, text: str) -> None:
    value = text.strip()
    if not value:
        return
    if _redis is not None:
        key = _key("history", chat_id, "user")
        await _redis.rpush(key, value)
        await _redis.ltrim(key, -CHAT_MEMORY_TURNS, -1)
        await _redis.expire(key, 24 * 60 * 60)
        return

    async with _state_lock:
        chat_user_history[chat_id].append(value)


async def save_assistant_message(chat_id: int, text: str) -> None:
    value = text.strip()
    if not value:
        return
    if _redis is not None:
        key = _key("history", chat_id, "assistant")
        await _redis.rpush(key, value)
        await _redis.ltrim(key, -CHAT_MEMORY_TURNS, -1)
        await _redis.expire(key, 24 * 60 * 60)
        return

    async with _state_lock:
        chat_assistant_history[chat_id].append(value)


async def get_user_history(chat_id: int, limit: int = 3) -> list[str]:
    if _redis is not None:
        values = await _redis.lrange(_key("history", chat_id, "user"), -limit, -1)
        return [str(v) for v in values]

    async with _state_lock:
        return list(chat_user_history.get(chat_id, []))[-limit:]


async def get_assistant_history(chat_id: int, limit: int = CHAT_MEMORY_TURNS) -> list[str]:
    if _redis is not None:
        values = await _redis.lrange(_key("history", chat_id, "assistant"), -limit, -1)
        return [str(v) for v in values]

    async with _state_lock:
        return list(chat_assistant_history.get(chat_id, []))[-limit:]


async def clear_chat_state(chat_id: int) -> None:
    if _redis is not None:
        await _redis.delete(
            _key("history", chat_id, "user"),
            _key("history", chat_id, "assistant"),
            _key("pending_calc", chat_id),
        )
        return

    async with _state_lock:
        chat_user_history.pop(chat_id, None)
        chat_assistant_history.pop(chat_id, None)
        chat_loan_calc_pending.pop(chat_id, None)


async def get_pending_calc(chat_id: int) -> dict[str, Any] | None:
    if _redis is not None:
        raw = await _redis.get(_key("pending_calc", chat_id))
        return json.loads(raw) if raw else None

    async with _state_lock:
        pending = chat_loan_calc_pending.get(chat_id)
        return dict(pending) if pending else None


async def set_pending_calc(chat_id: int, payload: dict[str, Any]) -> None:
    if _redis is not None:
        await _redis.set(_key("pending_calc", chat_id), json.dumps(payload), ex=30 * 60)
        return

    async with _state_lock:
        chat_loan_calc_pending[chat_id] = dict(payload)


async def clear_pending_calc(chat_id: int) -> None:
    if _redis is not None:
        await _redis.delete(_key("pending_calc", chat_id))
        return

    async with _state_lock:
        chat_loan_calc_pending.pop(chat_id, None)


async def is_chat_rate_limited(chat_id: int) -> bool:
    if RATE_LIMIT_MAX_MESSAGES <= 0:
        return False

    now = time.monotonic()
    if _redis is not None:
        key = _key("rate", "chat", chat_id)
        wall_now = time.time()
        await _redis.zremrangebyscore(key, 0, wall_now - RATE_LIMIT_WINDOW_SECONDS)
        await _redis.zadd(key, {str(wall_now): wall_now})
        await _redis.expire(key, RATE_LIMIT_WINDOW_SECONDS + 60)
        return await _redis.zcard(key) > RATE_LIMIT_MAX_MESSAGES

    async with _state_lock:
        window = chat_message_windows[chat_id]
        _prune_old_timestamps(window, now, RATE_LIMIT_WINDOW_SECONDS)
        window.append(now)
        return len(window) > RATE_LIMIT_MAX_MESSAGES


async def can_send_rate_limit_notice(chat_id: int) -> bool:
    now = time.monotonic()
    if _redis is not None:
        key = _key("rate_notice", chat_id)
        was_set = await _redis.set(key, "1", nx=True, ex=max(RATE_LIMIT_NOTICE_COOLDOWN_SECONDS, 1))
        return bool(was_set)

    async with _state_lock:
        last_notice_at = chat_last_rate_limit_notice_at.get(chat_id, 0.0)
        if now - last_notice_at < RATE_LIMIT_NOTICE_COOLDOWN_SECONDS:
            return False
        chat_last_rate_limit_notice_at[chat_id] = now
        return True


async def reserve_openai_call(chat_id: int) -> tuple[bool, str | None]:
    """Return False before an OpenAI call if the call should be blocked."""
    now = time.monotonic()
    if _redis is not None:
        if OPENAI_COOLDOWN_SECONDS > 0:
            cooldown_key = _key("openai", "cooldown", chat_id)
            was_set = await _redis.set(cooldown_key, "1", nx=True, ex=max(int(OPENAI_COOLDOWN_SECONDS), 1))
            if not was_set:
                return False, "chat_cooldown"

        if OPENAI_GLOBAL_MAX_CALLS > 0:
            key = _key("openai", "global_window")
            wall_now = time.time()
            await _redis.zremrangebyscore(key, 0, wall_now - OPENAI_GLOBAL_WINDOW_SECONDS)
            count = await _redis.zcard(key)
            if count >= OPENAI_GLOBAL_MAX_CALLS:
                return False, "global_openai_limit"
            await _redis.zadd(key, {f"{chat_id}:{wall_now}": wall_now})
            await _redis.expire(key, OPENAI_GLOBAL_WINDOW_SECONDS + 60)
        return True, None

    async with _state_lock:
        last_call_at = chat_last_openai_at.get(chat_id, 0.0)
        if OPENAI_COOLDOWN_SECONDS > 0 and now - last_call_at < OPENAI_COOLDOWN_SECONDS:
            return False, "chat_cooldown"

        if OPENAI_GLOBAL_MAX_CALLS > 0:
            _prune_old_timestamps(global_openai_window, now, OPENAI_GLOBAL_WINDOW_SECONDS)
            if len(global_openai_window) >= OPENAI_GLOBAL_MAX_CALLS:
                return False, "global_openai_limit"
            global_openai_window.append(now)

        chat_last_openai_at[chat_id] = now
        return True, None
