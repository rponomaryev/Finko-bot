import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram-ai-bot")


class Settings(BaseSettings):
    openai_api_key: str
    telegram_bot_token: str
    telegram_webhook_secret: str
    openai_vector_store_id: str

    openai_model: str = "gpt-4.1-mini"
    kb_search_results: int = 20
    kb_max_chunks: int = 8
    chat_memory_turns: int = 8

    telegram_timeout_seconds: float = 30
    telegram_max_retries: int = 3
    telegram_retry_base_delay: float = 0.5
    mini_app_url: str = ""
    mini_app_button_text: str = "Подобрать кредит"

    openai_max_retries: int = 3
    openai_retry_base_delay: float = 0.5

    # Optional. If set on Railway, conversational memory, pending calculator
    # state, duplicate update cache, and rate limits are shared across restarts
    # and multiple workers. Without it the bot falls back to in-memory state.
    redis_url: str = ""
    redis_key_prefix: str = "finko_bot"

    database_path: str = str(BASE_DIR / "bot_analytics.db")
    admin_analytics_token: str = ""

    max_webhook_body_bytes: int = 262144
    max_message_chars: int = 2000
    processed_updates_maxlen: int = 2000

    rate_limit_window_seconds: int = 60
    rate_limit_max_messages: int = 20
    rate_limit_notice_cooldown_seconds: int = 60
    openai_cooldown_seconds: float = 3
    openai_global_window_seconds: int = 60
    openai_global_max_calls: int = 60

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

OPENAI_API_KEY = settings.openai_api_key
TELEGRAM_BOT_TOKEN = settings.telegram_bot_token
TELEGRAM_WEBHOOK_SECRET = settings.telegram_webhook_secret
OPENAI_MODEL = settings.openai_model
OPENAI_VECTOR_STORE_ID = settings.openai_vector_store_id

KB_SEARCH_RESULTS = settings.kb_search_results
KB_MAX_CHUNKS = settings.kb_max_chunks
CHAT_MEMORY_TURNS = settings.chat_memory_turns
TELEGRAM_TIMEOUT_SECONDS = settings.telegram_timeout_seconds
TELEGRAM_MAX_RETRIES = settings.telegram_max_retries
TELEGRAM_RETRY_BASE_DELAY = settings.telegram_retry_base_delay
MINI_APP_URL = settings.mini_app_url.strip()
MINI_APP_BUTTON_TEXT = settings.mini_app_button_text.strip()
OPENAI_MAX_RETRIES = settings.openai_max_retries
OPENAI_RETRY_BASE_DELAY = settings.openai_retry_base_delay
REDIS_URL = settings.redis_url.strip()
REDIS_KEY_PREFIX = settings.redis_key_prefix.strip() or "finko_bot"

DATABASE_PATH = settings.database_path
ADMIN_ANALYTICS_TOKEN = settings.admin_analytics_token

MAX_WEBHOOK_BODY_BYTES = settings.max_webhook_body_bytes
MAX_MESSAGE_CHARS = settings.max_message_chars
PROCESSED_UPDATES_MAXLEN = settings.processed_updates_maxlen
RATE_LIMIT_WINDOW_SECONDS = settings.rate_limit_window_seconds
RATE_LIMIT_MAX_MESSAGES = settings.rate_limit_max_messages
RATE_LIMIT_NOTICE_COOLDOWN_SECONDS = settings.rate_limit_notice_cooldown_seconds
OPENAI_COOLDOWN_SECONDS = settings.openai_cooldown_seconds
OPENAI_GLOBAL_WINDOW_SECONDS = settings.openai_global_window_seconds
OPENAI_GLOBAL_MAX_CALLS = settings.openai_global_max_calls
