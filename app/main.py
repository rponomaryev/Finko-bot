import logging
import os
import re
import sqlite3
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from openai import OpenAI, RateLimitError

# ============================================================
# Environment / config
# ============================================================

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

KB_SEARCH_RESULTS = int(os.getenv("KB_SEARCH_RESULTS", "20"))
KB_MAX_CHUNKS = int(os.getenv("KB_MAX_CHUNKS", "8"))
CHAT_MEMORY_TURNS = int(os.getenv("CHAT_MEMORY_TURNS", "8"))
TELEGRAM_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "30"))

DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "bot_analytics.db"))
ADMIN_ANALYTICS_TOKEN = os.getenv("ADMIN_ANALYTICS_TOKEN", "")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not TELEGRAM_WEBHOOK_SECRET:
    raise RuntimeError("TELEGRAM_WEBHOOK_SECRET is not set")
if not OPENAI_VECTOR_STORE_ID:
    raise RuntimeError("OPENAI_VECTOR_STORE_ID is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Telegram AI Bot", version="4.2.0")

# ============================================================
# In-memory state
# ============================================================

processed_updates: set[int] = set()

chat_user_history: dict[int, deque[str]] = defaultdict(
    lambda: deque(maxlen=CHAT_MEMORY_TURNS)
)
chat_assistant_history: dict[int, deque[str]] = defaultdict(
    lambda: deque(maxlen=5)
)

db_lock = threading.Lock()

# ============================================================
# Database
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db_lock:
        with get_db() as conn:
            conn.executescript(
                """
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


@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized at %s", DATABASE_PATH)


def upsert_user_profile(
    chat_id: int,
    username: str | None,
    first_name: str | None,
    last_name: str | None,
    language: str,
    user_type: str,
) -> None:
    now = utc_now_iso()
    with db_lock:
        with get_db() as conn:
            row = conn.execute(
                "SELECT chat_id FROM users WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()

            if row:
                conn.execute(
                    """
                    UPDATE users
                    SET username = ?,
                        first_name = ?,
                        last_name = ?,
                        last_language = ?,
                        user_type = ?,
                        last_seen_at = ?,
                        messages_count = messages_count + 1
                    WHERE chat_id = ?
                    """,
                    (
                        username,
                        first_name,
                        last_name,
                        language,
                        user_type,
                        now,
                        chat_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO users (
                        chat_id, username, first_name, last_name,
                        last_language, user_type,
                        first_seen_at, last_seen_at, messages_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        chat_id,
                        username,
                        first_name,
                        last_name,
                        language,
                        user_type,
                        now,
                        now,
                    ),
                )


def log_message(
    chat_id: int,
    direction: str,
    text: str,
    language: str | None,
    intent: str | None,
    user_type: str | None,
    source: str,
) -> None:
    with db_lock:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    chat_id, direction, text, language, intent, user_type, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    direction,
                    text,
                    language,
                    intent,
                    user_type,
                    source,
                    utc_now_iso(),
                ),
            )


def log_event(chat_id: int | None, event_type: str, payload: str | None = None) -> None:
    with db_lock:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO events (chat_id, event_type, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, event_type, payload, utc_now_iso()),
            )

# ============================================================
# Language detection
# ============================================================

UZ_CYR_SPECIFIC_RE = re.compile(r"[ЎўҚқҒғҲҳ]")
ANY_CYR_RE = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳ]")
LATIN_RE = re.compile(r"[A-Za-z]")

UZ_LATN_HINTS_RE = re.compile(
    r"\b("
    r"yo'q|ha|salom|assalomu|rahmat|iltimos|bo'yicha|qanday|qanaqa|mumkin|kerak|"
    r"ariza|banklar|hamkorlar|mijoz|foiz|muddat|shartlar|tasdiq|rad|hujjat|"
    r"to'lov|o'zbek|uzbek|qaysi|bilan|ishlaysiz|kompaniya|tashkilot|"
    r"ma'lumot|mavjud|kerakmi|bormi|qiladi|qilinadi|yoki|hamkor|biznes|"
    r"aloqa|kontakt|kredit|qarz|mfo|mmt|moliya|savol|javob|foydalanuvchi|"
    r"kreditlar|kontaktlar|hamkorlarga|biznes"
    r")\b",
    re.IGNORECASE,
)

UZ_CYRL_HINTS_RE = re.compile(
    r"\b("
    r"салом|ассалому|рахмат|илтимос|қандай|қайси|мумкин|керак|"
    r"ариза|банклар|ҳамкорлар|мижоз|фоиз|муддат|шартлар|ҳужжат|"
    r"тўлов|маълумот|мавжуд|алоқа|контакт|кредит|қарз|ммт|молия|"
    r"жавоб|савол|фойдаланувчи|контактлар|кредитлар|ҳамкорларга|бизнес"
    r")\b",
    re.IGNORECASE,
)

EN_HINTS_RE = re.compile(
    r"\b("
    r"hello|hi|thanks|please|loan|credit|application|status|bank|banks|"
    r"partner|partners|insurance|leasing|how|what|where|when|can|do|does|"
    r"is|are|which|work|with|company|information|available|customer|"
    r"business|contact|contacts|support|help|mortgage|microfinance"
    r")\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """
    Returns one of:
    - ru
    - uz_latn
    - uz_cyrl
    - en
    """
    lowered = text.lower().strip()

    if not lowered:
        return "ru"

    if UZ_CYR_SPECIFIC_RE.search(text):
        return "uz_cyrl"

    if ANY_CYR_RE.search(text):
        if UZ_CYRL_HINTS_RE.search(lowered):
            return "uz_cyrl"
        return "ru"

    uz_score = len(UZ_LATN_HINTS_RE.findall(lowered))
    en_score = len(EN_HINTS_RE.findall(lowered))

    if any(x in lowered for x in ["o'", "g'", "yo'q", "ya'ni"]):
        uz_score += 3

    if en_score > uz_score:
        return "en"
    if uz_score > en_score:
        return "uz_latn"

    if LATIN_RE.search(text):
        common_en = {"the", "what", "which", "hello", "hi", "bank", "banks", "contacts"}
        if any(word in lowered.split() for word in common_en):
            return "en"
        return "uz_latn"

    return "ru"


def lang_name(lang: str) -> str:
    return {
        "ru": "Russian",
        "uz_latn": "Uzbek written in Latin script",
        "uz_cyrl": "Uzbek written in Cyrillic script",
        "en": "English",
    }.get(lang, "Russian")


def not_found_message(lang: str) -> str:
    messages = {
        "ru": "Я не нашёл точную информацию в базе знаний FINKO.",
        "uz_latn": "FINKO bilimlar bazasida aniq ma'lumot topilmadi.",
        "uz_cyrl": "FINKO билимлар базасида аниқ маълумот топилмади.",
        "en": "I could not find exact information in the FINKO knowledge base.",
    }
    return messages.get(lang, messages["ru"])


def server_error_message(lang: str) -> str:
    messages = {
        "ru": "Произошла ошибка на сервере. Попробуйте чуть позже.",
        "uz_latn": "Serverda xatolik yuz berdi. Iltimos, biroz keyinroq urinib ko'ring.",
        "uz_cyrl": "Серверда хатолик юз берди. Илтимос, бироз кейинроқ уриниб кўринг.",
        "en": "A server error occurred. Please try again a little later.",
    }
    return messages.get(lang, messages["ru"])


def quota_error_message(lang: str) -> str:
    messages = {
        "ru": "OpenAI API временно недоступен: проверьте квоту и billing.",
        "uz_latn": "OpenAI API vaqtincha ishlamayapti: kvota va billingni tekshiring.",
        "uz_cyrl": "OpenAI API вақтинча ишламаяпти: квота ва billingни текширинг.",
        "en": "The OpenAI API is temporarily unavailable: please check quota and billing.",
    }
    return messages.get(lang, messages["ru"])

# ============================================================
# Intents / user types
# ============================================================

def normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ0-9']+", text.lower())
    stopwords = {
        "и", "в", "на", "по", "с", "у", "a", "the", "is", "are", "to", "for",
        "va", "ham", "bilan", "qanday", "what", "how", "please", "привет",
        "здравствуйте", "salom", "hello", "hi", "ва", "билан", "қандай",
    }
    return {w for w in words if len(w) > 2 and w not in stopwords}


def text_similarity(a: str, b: str) -> float:
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union else 0.0


def detect_intent(user_text: str) -> str:
    text = normalize_text_for_match(user_text)

    if text in {
        "/start", "/restart", "restart", "перезапуск", "qayta", "qayta ishga tushirish",
        "қайта", "қайта ишга тушириш"
    }:
        return "restart"

    if text in {
        "kontakti", "kontaktlar", "contacts", "contact",
        "контакты", "контакт", "контактлар", "алоқа", "/contacts"
    }:
        return "contacts"

    if text in {
        "кредиты", "kreditlar", "credits", "credit", "loan", "loans", "/credits",
        "кредитлар"
    }:
        return "credits_menu"

    if text in {
        "бизнес", "biznes", "business", "/business"
    }:
        return "business_menu"

    if text in {
        "партнёры", "партнеры", "партнёрам", "партнерам",
        "hamkorlar", "hamkorlarga", "partners", "partner", "/partners",
        "ҳамкорлар", "ҳамкорларга"
    }:
        return "partners_menu"

    if text in {"привет", "здравствуйте", "салом", "salom", "hello", "hi", "assalomu alaykum", "ассалому алайкум"}:
        return "greeting"

    if text in {"спасибо", "rahmat", "thanks", "thank you", "рахмат"}:
        return "thanks"

    partner_keywords = [
        "партнер", "партнёр", "hamkor", "partner", "bank", "mfo", "мфо",
        "api", "integration", "интегра", "erp", "подключ", "onboarding",
        "банк", "лизинг", "страхов", "mmt", "hamkorlar",
        "ҳамкор", "банклар", "ммт", "суғурта",
    ]
    business_keywords = [
        "business", "biznes", "бизнес", "компания", "company", "corporate",
        "юрид", "yuridik", "ип", "ooo", "llc", "предприним", "biznes kredit",
        "бизнес кредит", "юридик", "компаниялар",
    ]
    credit_keywords = [
        "кредит", "credit", "loan", "ипотек", "mortgage", "авто", "car loan",
        "microloan", "mikrozaym", "микроз", "qarz", "kredit", "ipoteka",
        "кредитлар", "қарз", "ипотека",
    ]
    contact_keywords = [
        "контакт", "contact", "contacts", "aloqa", "телефон", "email", "почта",
        "support", "поддерж", "help", "контактлар", "алоқа",
    ]

    if any(k in text for k in contact_keywords):
        return "contacts"

    if any(k in text for k in partner_keywords):
        return "partners"

    if any(k in text for k in business_keywords):
        return "business"

    if any(k in text for k in credit_keywords):
        return "credits"

    return "general"


def infer_user_type(intent: str, user_text: str) -> str:
    text = normalize_text_for_match(user_text)

    if intent in {"partners", "partners_menu"}:
        return "partner"

    if intent in {"business", "business_menu"}:
        return "business"

    if any(k in text for k in ["bank", "банк", "mfo", "мфо", "partner", "hamkor", "api", "erp", "ҳамкор"]):
        return "partner"

    if any(k in text for k in ["company", "компания", "yuridik", "юрид", "biznes", "бизнес", "ип", "юридик"]):
        return "business"

    return "customer"

# ============================================================
# Telegram UI
# ============================================================

def get_keyboard_for_lang(lang: str) -> dict[str, Any]:
    labels = {
        "ru": {
            "credits": "Кредиты",
            "business": "Бизнес",
            "partners": "Партнёры",
            "restart": "Restart",
            "contacts": "Контакты",
            "placeholder": "Напишите вопрос...",
        },
        "uz_latn": {
            "credits": "Kreditlar",
            "business": "Biznes",
            "partners": "Hamkorlar",
            "restart": "Restart",
            "contacts": "Kontaktlar",
            "placeholder": "Savolingizni yozing...",
        },
        "uz_cyrl": {
            "credits": "Кредитлар",
            "business": "Бизнес",
            "partners": "Ҳамкорлар",
            "restart": "Restart",
            "contacts": "Контактлар",
            "placeholder": "Саволингизни ёзинг...",
        },
        "en": {
            "credits": "Credits",
            "business": "Business",
            "partners": "Partners",
            "restart": "Restart",
            "contacts": "Contacts",
            "placeholder": "Write a message...",
        },
    }[lang]

    return {
        "keyboard": [
            [{"text": labels["credits"]}, {"text": labels["business"]}],
            [{"text": labels["partners"]}],
            [{"text": labels["restart"]}, {"text": labels["contacts"]}],
        ],
        "resize_keyboard": True,
        "persistent_keyboard": True,
        "input_field_placeholder": labels["placeholder"],
    }


async def send_telegram_message(chat_id: int, text: str, ui_lang: str = "ru") -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": get_keyboard_for_lang(ui_lang),
    }

    async with httpx.AsyncClient(timeout=TELEGRAM_TIMEOUT_SECONDS) as http_client:
        response = await http_client.post(url, json=payload)
        response.raise_for_status()


def extract_user_message(update: dict[str, Any]) -> tuple[int | None, str | None, dict[str, Any]]:
    message = update.get("message")
    if not message:
        return None, None, {}

    chat = message.get("chat", {})
    from_user = message.get("from", {})

    chat_id = chat.get("id")
    text = message.get("text")

    if not chat_id or not text:
        return None, None, {}

    profile = {
        "username": from_user.get("username"),
        "first_name": from_user.get("first_name"),
        "last_name": from_user.get("last_name"),
    }

    return chat_id, text.strip(), profile

# ============================================================
# Quick answers without OpenAI
# ============================================================

def build_quick_answer(action: str, lang: str) -> str:
    answers = {
        "restart": {
            "ru": "Бот перезапущен. Можете отправить новый вопрос.",
            "uz_latn": "Bot qayta ishga tushdi. Yangi savol yuborishingiz mumkin.",
            "uz_cyrl": "Бот қайта ишга тушди. Янги савол юборишингиз мумкин.",
            "en": "The bot has been restarted. You can send a new question.",
        },
        "contacts": {
            "ru": (
                "Контакты FINKO:\n"
                "Телефон: +998 50 177 77 88\n"
                "Email: info@finko.uz\n"
                "Сайт: https://finko.uz\n"
                "Офис: Ташкент, ул. Ойбек 18/1, БЦ ATRIUM"
            ),
            "uz_latn": (
                "FINKO kontaktlari:\n"
                "Telefon: +998 50 177 77 88\n"
                "Email: info@finko.uz\n"
                "Sayt: https://finko.uz\n"
                "Ofis: Toshkent, Oybek 18/1, ATRIUM"
            ),
            "uz_cyrl": (
                "FINKO контактлари:\n"
                "Телефон: +998 50 177 77 88\n"
                "Email: info@finko.uz\n"
                "Сайт: https://finko.uz\n"
                "Офис: Тошкент, Ойбек 18/1, ATRIUM"
            ),
            "en": (
                "FINKO contacts:\n"
                "Phone: +998 50 177 77 88\n"
                "Email: info@finko.uz\n"
                "Website: https://finko.uz\n"
                "Office: Tashkent, Oybek 18/1, ATRIUM"
            ),
        },
        "credits_menu": {
            "ru": (
                "Через FINKO доступны потребительские кредиты, автокредиты, ипотека, "
                "микрозаймы и другие финансовые продукты. Условия по сумме, сроку и ставке "
                "определяются банком или МФО-партнёром. FINKO не выдаёт кредиты напрямую."
            ),
            "uz_latn": (
                "FINKO orqali iste'mol kreditlari, avtokreditlar, ipoteka, mikrozaymlar "
                "va boshqa moliyaviy mahsulotlar mavjud. Summa, muddat va stavka hamkor bank "
                "yoki MMT tomonidan belgilanadi. FINKO kreditni to'g'ridan-to'g'ri bermaydi."
            ),
            "uz_cyrl": (
                "FINKO орқали истеъмол кредитлари, автокредитлар, ипотека, микрозаймлар "
                "ва бошқа молиявий маҳсулотлар мавжуд. Сумма, муддат ва ставка ҳамкор банк "
                "ёки ММТ томонидан белгиланади. FINKO кредитни тўғридан-тўғри бермайди."
            ),
            "en": (
                "Through FINKO, users can access consumer loans, auto loans, mortgages, "
                "microloans, and other financial products. The amount, term, and rate are "
                "set by the partner bank or MFO. FINKO does not issue loans directly."
            ),
        },
        "business_menu": {
            "ru": (
                "Для бизнеса через FINKO доступны бизнес-кредиты, оборотные и инвестиционные "
                "кредиты, лизинг, страхование и другие решения. Итоговые условия определяются "
                "партнёрской организацией."
            ),
            "uz_latn": (
                "Biznes uchun FINKO orqali biznes kreditlari, aylanma va investitsiya "
                "kreditlari, lizing, sug'urta va boshqa yechimlar mavjud. Yakuniy shartlar "
                "hamkor tashkilot tomonidan belgilanadi."
            ),
            "uz_cyrl": (
                "Бизнес учун FINKO орқали бизнес кредитлари, айланма ва инвестиция "
                "кредитлари, лизинг, суғурта ва бошқа ечимлар мавжуд. Якуний шартлар "
                "ҳамкор ташкилот томонидан белгиланади."
            ),
            "en": (
                "For businesses, FINKO offers access to business loans, working capital and "
                "investment loans, leasing, insurance, and related solutions. Final terms are "
                "set by the partner organization."
            ),
        },
        "partners_menu": {
            "ru": (
                "FINKO сотрудничает с Hamkorbank, Universal Bank, Davr-Bank и Madad Invest Bank, "
                "а также с МФО DELTA, PULMAN, APEX MOLIYA, ANSOR и ALMIZAN. Количество партнёрских "
                "организаций постоянно увеличивается."
            ),
            "uz_latn": (
                "FINKO Hamkorbank, Universal Bank, Davr-Bank va Madad Invest Bank bilan, "
                "shuningdek DELTA, PULMAN, APEX MOLIYA, ANSOR va ALMIZAN kabi MMTlar bilan "
                "hamkorlik qiladi. Hamkor tashkilotlar soni doimiy ravishda oshib bormoqda."
            ),
            "uz_cyrl": (
                "FINKO Hamkorbank, Universal Bank, Davr-Bank ва Madad Invest Bank билан, "
                "шунингдек DELTA, PULMAN, APEX MOLIYA, ANSOR ва ALMIZAN каби ММТлар билан "
                "ҳамкорлик қилади. Ҳамкор ташкилотлар сони доимий равишда ошиб бормоқда."
            ),
            "en": (
                "FINKO works with Hamkorbank, Universal Bank, Davr-Bank, and Madad Invest Bank, "
                "as well as the MFOs DELTA, PULMAN, APEX MOLIYA, ANSOR, and ALMIZAN. The number "
                "of partner organizations is continuously growing."
            ),
        },
        "greeting": {
            "ru": "Здравствуйте! Я AI-ассистент FINKO. Могу помочь с продуктами, бизнес-вопросами, партнёрством и контактами.",
            "uz_latn": "Salom! Men FINKO AI yordamchisiman. Mahsulotlar, biznes savollari, hamkorlik va kontaktlar bo'yicha yordam bera olaman.",
            "uz_cyrl": "Салом! Мен FINKO AI ёрдамчисиман. Маҳсулотлар, бизнес саволлари, ҳамкорлик ва контактлар бўйича ёрдам бера оламан.",
            "en": "Hello! I’m the FINKO AI assistant. I can help with products, business questions, partnerships, and contacts.",
        },
        "thanks": {
            "ru": "Пожалуйста! Если захотите, можете задать ещё один вопрос.",
            "uz_latn": "Marhamat! Xohlasangiz, yana savol yuborishingiz mumkin.",
            "uz_cyrl": "Марҳамат! Хоҳласангиз, яна савол юборишингиз мумкин.",
            "en": "You’re welcome! Feel free to send another question.",
        },
    }

    if action not in answers:
        return ""

    return answers[action].get(lang, answers[action]["ru"])


def should_use_quick_reply(intent: str, user_text: str) -> bool:
    text = normalize_text_for_match(user_text)

    if intent in {
        "restart", "contacts", "credits_menu", "business_menu", "partners_menu",
        "greeting", "thanks"
    }:
        return True

    if len(text.split()) <= 3 and intent in {"credits", "business", "partners"}:
        return True

    return False


def handle_menu_or_quick_action(user_text: str, chat_id: int) -> tuple[str | None, str | None]:
    lang = detect_language(user_text)
    intent = detect_intent(user_text)

    if not should_use_quick_reply(intent, user_text):
        return None, intent

    if intent == "restart":
        chat_user_history.pop(chat_id, None)
        chat_assistant_history.pop(chat_id, None)

    quick_intent = intent
    if intent == "credits":
        quick_intent = "credits_menu"
    elif intent == "business":
        quick_intent = "business_menu"
    elif intent == "partners":
        quick_intent = "partners_menu"

    answer = build_quick_answer(quick_intent, lang)
    return answer or None, intent

# ============================================================
# KB search
# ============================================================

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
    lowered = user_text.lower().strip()

    if intent in {"partners", "partners_menu"}:
        if detected_lang == "ru":
            queries.extend([
                "банки партнеры FINKO",
                "Hamkorbank Universal Bank Davr-Bank Madad Invest Bank",
                "МФО партнеры FINKO DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN",
            ])
        elif detected_lang in {"uz_latn", "uz_cyrl"}:
            queries.extend([
                "FINKO hamkor banklar",
                "Hamkorbank Universal Bank Davr-Bank Madad Invest Bank",
                "FINKO hamkor MMTlar DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN",
                "FINKO ҳамкор банклар",
                "FINKO ҳамкор ММТлар DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN",
            ])
        else:
            queries.extend([
                "FINKO partner banks",
                "Hamkorbank Universal Bank Davr-Bank Madad Invest Bank",
                "FINKO partner MFOs DELTA PULMAN APEX MOLIYA ANSOR ALMIZAN",
            ])

    elif intent in {"business", "business_menu"}:
        if detected_lang == "ru":
            queries.extend([
                "бизнес кредиты FINKO",
                "лизинг страхование бизнес FINKO",
            ])
        elif detected_lang in {"uz_latn", "uz_cyrl"}:
            queries.extend([
                "FINKO biznes kreditlari",
                "lizing sug'urta biznes FINKO",
                "FINKO бизнес кредитлари",
                "лизинг суғурта бизнес FINKO",
            ])
        else:
            queries.extend([
                "FINKO business loans",
                "FINKO leasing insurance business",
            ])

    elif intent in {"credits", "credits_menu"}:
        if detected_lang == "ru":
            queries.extend([
                "кредиты FINKO",
                "ипотека автокредит микрозаймы FINKO",
            ])
        elif detected_lang in {"uz_latn", "uz_cyrl"}:
            queries.extend([
                "FINKO kreditlar",
                "ipoteka avtokredit mikrozaym FINKO",
                "FINKO кредитлар",
                "ипотека автокредит микрозайм FINKO",
            ])
        else:
            queries.extend([
                "FINKO credits",
                "mortgage auto loan microloans FINKO",
            ])

    if detected_lang == "ru":
        if "какие банки" in lowered or "с какими банками" in lowered:
            queries.append("банки партнеры FINKO")
        elif "мфо" in lowered or "микрофинансов" in lowered:
            queries.append("МФО партнеры FINKO")
    elif detected_lang in {"uz_latn", "uz_cyrl"}:
        if (
            "qaysi banklar" in lowered
            or "banklar bilan" in lowered
            or "қайси банклар" in lowered
            or "банклар билан" in lowered
        ):
            queries.extend(["FINKO hamkor banklar", "FINKO ҳамкор банклар"])
        elif (
            "mmt" in lowered
            or "mikromoliyaviy" in lowered
            or "ммт" in lowered
            or "микромолиявий" in lowered
        ):
            queries.extend(["FINKO hamkor MMTlar", "FINKO ҳамкор ММТлар"])
    elif detected_lang == "en":
        if "which banks" in lowered or "banks do you work with" in lowered:
            queries.append("FINKO partner banks")
        elif "mfo" in lowered or "microfinance" in lowered:
            queries.append("FINKO partner MFOs")

    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(query)

    return unique[:7]


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

    language_priority = preferred_search_languages(preferred_lang)
    grouped_chunks: dict[str, list[str]] = {lang: [] for lang in language_priority}
    grouped_chunks["other"] = []

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


def search_knowledge_base(user_text: str, preferred_lang: str, intent: str) -> str | None:
    queries = build_search_queries(user_text, preferred_lang, intent)

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

# ============================================================
# Repetition control
# ============================================================

def get_history_block(chat_id: int) -> str:
    messages = chat_user_history.get(chat_id)
    if not messages:
        return "No previous user messages."

    recent = list(messages)[-3:]
    return "\n".join(f"User: {text}" for text in recent)


def save_user_message(chat_id: int, text: str) -> None:
    chat_user_history[chat_id].append(text.strip())


def save_assistant_message(chat_id: int, text: str) -> None:
    chat_assistant_history[chat_id].append(text.strip())


def is_semantically_repeated_question(chat_id: int, user_text: str) -> bool:
    history = chat_user_history.get(chat_id)
    if not history:
        return False

    recent = list(history)[-3:]
    for old_text in recent:
        if text_similarity(old_text, user_text) >= 0.85:
            return True
    return False


def reduce_repetition_if_needed(chat_id: int, answer: str) -> str:
    recent_answers = chat_assistant_history.get(chat_id)
    if not recent_answers:
        return answer

    for old_answer in recent_answers:
        if text_similarity(old_answer, answer) >= 0.86:
            sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
            if len(sentences) >= 2:
                return " ".join(sentences[:2]).strip()
            return answer

    return answer

# ============================================================
# Prompting / LLM
# ============================================================

def build_system_prompt(response_language: str, user_type: str) -> str:
    return f"""
You are the official AI assistant of FINKO.

CRITICAL LANGUAGE RULE:
- The language of the latest user message is {response_language}.
- You must answer only in {response_language}.
- Ignore the language used earlier in the conversation.
- Ignore the language of previous assistant replies.
- Never continue in a previous language unless the latest user message is in that language.
- Never mix Russian, Uzbek, and English in one reply unless the user explicitly asks for translation.
- If the language is Uzbek written in Latin script, answer Uzbek in Latin script only.
- If the language is Uzbek written in Cyrillic script, answer Uzbek in Cyrillic script only.

USER TYPE:
- Detected user type: {user_type}.
- Adapt phrasing to this user type while staying concise.

KNOWLEDGE RULES:
- Use only the provided FINKO knowledge-base context.
- Do not invent facts, rates, limits, approvals, partner conditions, or timelines.
- If the exact answer is not present in the context, say so honestly.
- Paraphrase naturally. Do not copy chunks verbatim.
- Keep the answer concise, clear, and natural.
- Avoid repeating the same information if it was already covered recently.
- If retrieved context is in another language or script, translate/transliterate its meaning into {response_language}.
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
    intent: str,
    repeated_question: bool,
) -> str:
    repeat_instruction = (
        "The user is asking a repeated or very similar question. Answer more briefly than before and avoid repeating wording."
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
- Answer only in {response_language}.
- Focus on the latest user message.
- Do not let earlier messages override the answer language.
- Be accurate and concise.
- Do not repeat the same facts unnecessarily.
- Do not copy the context word-for-word.
- {repeat_instruction}
- If the context is insufficient, say that exact information is not available in the FINKO knowledge base.
""".strip()


def generate_answer(chat_id: int, user_text: str, intent: str, user_type: str) -> str:
    response_lang = detect_language(user_text)

    kb_context = search_knowledge_base(
        user_text=user_text,
        preferred_lang=response_lang,
        intent=intent,
    )

    if not kb_context:
        return not_found_message(response_lang)

    history_block = get_history_block(chat_id)
    repeated_question = is_semantically_repeated_question(chat_id, user_text)

    prompt = build_user_prompt(
        user_text=user_text,
        response_language=lang_name(response_lang),
        kb_context=kb_context,
        history_block=history_block,
        intent=intent,
        repeated_question=repeated_question,
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=build_system_prompt(lang_name(response_lang), user_type),
            input=prompt,
            temperature=0.2,
        )

        answer = (response.output_text or "").strip()

        if not answer:
            return not_found_message(response_lang)

        answer = reduce_repetition_if_needed(chat_id, answer)
        return answer

    except RateLimitError:
        logger.exception("OpenAI quota exceeded")
        return quota_error_message(response_lang)

    except Exception as e:
        logger.exception("OpenAI answer generation failed: %s", e)
        return server_error_message(response_lang)

# ============================================================
# Analytics endpoints
# ============================================================

def verify_admin_token(x_admin_token: str | None) -> None:
    if not ADMIN_ANALYTICS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ADMIN_ANALYTICS_TOKEN is not configured",
        )
    if x_admin_token != ADMIN_ANALYTICS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token",
        )


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Telegram AI Bot is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/analytics/summary")
async def analytics_summary(
    x_admin_token: str | None = Header(default=None),
):
    verify_admin_token(x_admin_token)

    with db_lock:
        with get_db() as conn:
            users_count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
            messages_count = conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
            inbound_count = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE direction = 'inbound'"
            ).fetchone()["c"]
            outbound_count = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE direction = 'outbound'"
            ).fetchone()["c"]

            top_intents_rows = conn.execute(
                """
                SELECT intent, COUNT(*) AS c
                FROM messages
                WHERE direction = 'inbound' AND intent IS NOT NULL
                GROUP BY intent
                ORDER BY c DESC
                LIMIT 10
                """
            ).fetchall()

            top_languages_rows = conn.execute(
                """
                SELECT language, COUNT(*) AS c
                FROM messages
                WHERE direction = 'inbound' AND language IS NOT NULL
                GROUP BY language
                ORDER BY c DESC
                """
            ).fetchall()

            user_types_rows = conn.execute(
                """
                SELECT user_type, COUNT(*) AS c
                FROM users
                GROUP BY user_type
                ORDER BY c DESC
                """
            ).fetchall()

    return {
        "users_count": users_count,
        "messages_count": messages_count,
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        "top_intents": [{row["intent"]: row["c"]} for row in top_intents_rows],
        "languages": [{row["language"]: row["c"]} for row in top_languages_rows],
        "user_types": [{row["user_type"]: row["c"]} for row in user_types_rows],
    }


@app.get("/analytics/top-questions")
async def analytics_top_questions(
    limit: int = Query(default=10, ge=1, le=50),
    x_admin_token: str | None = Header(default=None),
):
    verify_admin_token(x_admin_token)

    with db_lock:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT text, COUNT(*) AS c
                FROM messages
                WHERE direction = 'inbound'
                GROUP BY text
                ORDER BY c DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return {
        "top_questions": [
            {"question": row["text"], "count": row["c"]}
            for row in rows
        ]
    }


@app.get("/analytics/top-users")
async def analytics_top_users(
    limit: int = Query(default=10, ge=1, le=50),
    x_admin_token: str | None = Header(default=None),
):
    verify_admin_token(x_admin_token)

    with db_lock:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT chat_id, username, first_name, last_name, messages_count, user_type, last_language
                FROM users
                ORDER BY messages_count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return {
        "top_users": [
            {
                "chat_id": row["chat_id"],
                "username": row["username"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "messages_count": row["messages_count"],
                "user_type": row["user_type"],
                "last_language": row["last_language"],
            }
            for row in rows
        ]
    }

# ============================================================
# Webhook
# ============================================================

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

    chat_id, user_text, profile = extract_user_message(update)
    if not chat_id or not user_text:
        return JSONResponse({"ok": True, "skipped": True})

    detected_lang = detect_language(user_text)
    intent = detect_intent(user_text)
    user_type = infer_user_type(intent, user_text)

    upsert_user_profile(
        chat_id=chat_id,
        username=profile.get("username"),
        first_name=profile.get("first_name"),
        last_name=profile.get("last_name"),
        language=detected_lang,
        user_type=user_type,
    )

    log_message(
        chat_id=chat_id,
        direction="inbound",
        text=user_text,
        language=detected_lang,
        intent=intent,
        user_type=user_type,
        source="telegram",
    )

    quick_answer, quick_intent = handle_menu_or_quick_action(user_text, chat_id)
    if quick_answer:
        await send_telegram_message(chat_id, quick_answer, ui_lang=detected_lang)
        save_assistant_message(chat_id, quick_answer)

        log_message(
            chat_id=chat_id,
            direction="outbound",
            text=quick_answer,
            language=detected_lang,
            intent=quick_intent,
            user_type=user_type,
            source="quick",
        )
        log_event(chat_id, "quick_reply", quick_intent)
        return JSONResponse({"ok": True, "source": "quick", "intent": quick_intent})

    try:
        save_user_message(chat_id, user_text)

        answer = generate_answer(
            chat_id=chat_id,
            user_text=user_text,
            intent=intent,
            user_type=user_type,
        )

        await send_telegram_message(chat_id, answer, ui_lang=detected_lang)
        save_assistant_message(chat_id, answer)

        log_message(
            chat_id=chat_id,
            direction="outbound",
            text=answer,
            language=detected_lang,
            intent=intent,
            user_type=user_type,
            source="openai",
        )
        log_event(chat_id, "openai_reply", intent)

        return JSONResponse(
            {
                "ok": True,
                "language": detected_lang,
                "intent": intent,
                "user_type": user_type,
                "source": "openai",
            }
        )

    except httpx.HTTPError as e:
        logger.exception("Telegram API error: %s", e)
        raise HTTPException(status_code=502, detail="Telegram API error")

    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        fallback_text = server_error_message(detected_lang)

        try:
            await send_telegram_message(chat_id, fallback_text, ui_lang=detected_lang)
            log_message(
                chat_id=chat_id,
                direction="outbound",
                text=fallback_text,
                language=detected_lang,
                intent=intent,
                user_type=user_type,
                source="fallback",
            )
            log_event(chat_id, "fallback_reply", str(e))
        except Exception:
            logger.exception("Failed to send fallback message")

        raise HTTPException(status_code=500, detail="Internal server error")