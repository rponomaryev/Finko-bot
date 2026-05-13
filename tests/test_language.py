from app.bot.language import detect_language
from app.bot.ui import get_keyboard_for_lang


def test_detects_russian():
    assert detect_language("Как получить кредит?") == "ru"


def test_detects_uzbek_latin():
    assert detect_language("Kredit olsam bo'ladimi?") == "uz_latn"


def test_detects_uzbek_cyrillic():
    assert detect_language("Кредит олиш мумкинми?") == "uz_cyrl"


def test_keyboard_fallback_for_unknown_language():
    keyboard = get_keyboard_for_lang("unknown")
    assert keyboard["keyboard"][0][0]["text"] == "Кредиты"
