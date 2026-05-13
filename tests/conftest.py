import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-telegram-token")
os.environ.setdefault("TELEGRAM_WEBHOOK_SECRET", "test-webhook-secret")
os.environ.setdefault("OPENAI_VECTOR_STORE_ID", "test-vector-store")
os.environ.setdefault("DATABASE_PATH", "/tmp/finko_bot_pytest.db")
os.environ.setdefault("REDIS_URL", "")
