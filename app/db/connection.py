import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import aiosqlite

from app.config import DATABASE_PATH

# aiosqlite is async, but schema migrations and writes are still serialized here
# to avoid SQLite write-contention surprises on a single Railway process.
db_lock = asyncio.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def get_db():
    conn = await aiosqlite.connect(DATABASE_PATH)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
        await conn.commit()
    finally:
        await conn.close()


async def ensure_column_exists(conn: aiosqlite.Connection, table: str, column: str, definition: str) -> None:
    async with conn.execute(f"PRAGMA table_info({table})") as cursor:
        rows = await cursor.fetchall()
    existing = {row["name"] for row in rows}
    if column not in existing:
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


async def init_db() -> None:
    async with db_lock:
        async with get_db() as conn:
            await conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA busy_timeout=5000;

                CREATE TABLE IF NOT EXISTS users (
                    chat_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    last_language TEXT,
                    user_type TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    messages_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT,
                    intent TEXT,
                    user_type TEXT,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_chat_created
                ON messages(chat_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_messages_intent
                ON messages(intent);

                CREATE INDEX IF NOT EXISTS idx_messages_language
                ON messages(language);
                """
            )

            await ensure_column_exists(conn, "users", "selected_language", "TEXT")
