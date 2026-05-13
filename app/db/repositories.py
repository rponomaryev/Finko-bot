import aiosqlite

from app.db.connection import db_lock, get_db, utc_now_iso

async def upsert_user_profile(
    chat_id: int,
    username: str | None,
    first_name: str | None,
    last_name: str | None,
    language: str,
    user_type: str,
    selected_language: str | None = None,
) -> None:
    now = utc_now_iso()
    async with db_lock:
        async with get_db() as conn:
            async with conn.execute(
                "SELECT chat_id FROM users WHERE chat_id = ?",
                (chat_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                fields = {
                    "username": username,
                    "first_name": first_name,
                    "last_name": last_name,
                    "last_language": language,
                    "user_type": user_type,
                    "last_seen_at": now,
                }
                if selected_language is not None:
                    fields["selected_language"] = selected_language

                set_clause = ", ".join(f"{key} = ?" for key in fields)
                await conn.execute(
                    f"""
                    UPDATE users
                    SET {set_clause},
                        messages_count = messages_count + 1
                    WHERE chat_id = ?
                    """,
                    (*fields.values(), chat_id),
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO users (
                        chat_id, username, first_name, last_name,
                        last_language, selected_language, user_type,
                        first_seen_at, last_seen_at, messages_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (chat_id, username, first_name, last_name, language, selected_language, user_type, now, now),
                )


async def get_user_ui_language(chat_id: int) -> str | None:
    async with db_lock:
        async with get_db() as conn:
            async with conn.execute(
                "SELECT selected_language FROM users WHERE chat_id = ?",
                (chat_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row and row["selected_language"]:
                return row["selected_language"]
    return None


async def log_message(
    chat_id: int,
    direction: str,
    text: str,
    language: str | None,
    intent: str | None,
    user_type: str | None,
    source: str,
) -> None:
    async with db_lock:
        async with get_db() as conn:
            await conn.execute(
                """
                INSERT INTO messages (
                    chat_id, direction, text, language, intent, user_type, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chat_id, direction, text, language, intent, user_type, source, utc_now_iso()),
            )


async def log_event(chat_id: int | None, event_type: str, payload: str | None = None) -> None:
    async with db_lock:
        async with get_db() as conn:
            await conn.execute(
                """
                INSERT INTO events (chat_id, event_type, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, event_type, payload, utc_now_iso()),
            )
