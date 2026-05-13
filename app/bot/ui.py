import re
from typing import Any

def get_keyboard_for_lang(lang: str) -> dict[str, Any]:
    labels_by_lang = {
        "ru": {
            "credits": "Кредиты",
            "business": "Бизнес",
            "partners": "Партнёры",
            "ask_question": "Задать вопрос",
            "contacts": "Контакты",
            "placeholder": "Напишите вопрос...",
        },
        "uz_latn": {
            "credits": "Kreditlar",
            "business": "Biznes",
            "partners": "Hamkorlar",
            "ask_question": "Savol yozish",
            "contacts": "Kontaktlar",
            "placeholder": "Savolingizni yozing...",
        },
        "uz_cyrl": {
            "credits": "Кредитлар",
            "business": "Бизнес",
            "partners": "Ҳамкорлар",
            "ask_question": "Савол ёзиш",
            "contacts": "Контактлар",
            "placeholder": "Саволингизни ёзинг...",
        },
        "en": {
            "credits": "Credits",
            "business": "Business",
            "partners": "Partners",
            "ask_question": "Ask a question",
            "contacts": "Contacts",
            "placeholder": "Write a message...",
        },
    }

    labels = labels_by_lang.get(lang, labels_by_lang["ru"])

    return {
        "keyboard": [
            [{"text": labels["credits"]}, {"text": labels["business"]}],
            [{"text": labels["partners"]}],
            [{"text": labels["ask_question"]}, {"text": labels["contacts"]}],
        ],
        "resize_keyboard": True,
        "persistent_keyboard": True,
        "input_field_placeholder": labels["placeholder"],
    }


def get_language_keyboard() -> dict[str, Any]:
    return {
        "keyboard": [
            [{"text": "🇷🇺 Русский"}],
            [{"text": "🇺🇿 O'zbek (Lotin)"}],
            [{"text": "🇺🇿 Ўзбек (Кирилл)"}],
            [{"text": "🇬🇧 English"}],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True,
        "input_field_placeholder": "Choose language / Выберите язык / Tilni tanlang",
    }


def handle_language_selection(text: str) -> str | None:
    lowered = text.lower().strip()

    if "рус" in lowered:
        return "ru"
    if "lotin" in lowered:
        return "uz_latn"
    if "кирил" in lowered or "ўзбек" in lowered:
        return "uz_cyrl"
    if "english" in lowered:
        return "en"

    return None


def build_language_saved_text(lang: str) -> str:
    messages = {
        "ru": "Язык интерфейса сохранён ✅",
        "uz_latn": "Interfeys tili saqlandi ✅",
        "uz_cyrl": "Интерфейс тили сақланди ✅",
        "en": "Interface language saved ✅",
    }
    return messages.get(lang, messages["ru"])


def build_start_language_text() -> str:
    return (
        "Выберите язык интерфейса.\n"
        "Choose interface language.\n"
        "Интерфейс тилини танланг."
    )


def build_language_clarification_text() -> str:
    return (
        "На каком языке вам удобно получить ответ?\n\n"
        "Русский / O'zbekcha (lotin) / Ўзбекча (кирилл) / English"
    )


def strip_cross_language_artifacts(answer: str, lang: str) -> str:
    if not answer:
        return answer

    cleaned = answer.strip()
    cleaned = re.sub(
        r"^(answer|response|ответ|javob|ж[ао]воб)\s*[:：]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    return cleaned
