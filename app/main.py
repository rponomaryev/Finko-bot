from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.analytics.routes import router as analytics_router
from app.bot.handlers import router as telegram_router
from app.bot.state import close_state_store, init_state_store
from app.config import DATABASE_PATH, logger
from app.db.connection import init_db
from app.services.telegram import close_telegram_client, init_telegram_client, set_telegram_menu_button


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Database initialized at %s", DATABASE_PATH)

    await init_state_store()
    await init_telegram_client()
    try:
        await set_telegram_menu_button()
        yield
    finally:
        await close_telegram_client()
        await close_state_store()


app = FastAPI(title="Telegram AI Bot", version="4.7.0", lifespan=lifespan)
app.include_router(analytics_router)
app.include_router(telegram_router)
