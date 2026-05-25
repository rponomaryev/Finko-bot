import pytest
from app.bot.language import detect_language, normalize_supported_lang, platform_link
from app.bot.ui import get_keyboard_for_lang, get_language_keyboard


# ── detect_language ──────────────────────────────────────────────────────────

def test_detects_russian():
    assert detect_language("Как получить кредит?") == "ru"


def test_detects_uzbek_latin():
    assert detect_language("Kredit olsam bo'ladimi?") == "uz_latn"


def test_detects_english():
    assert detect_language("How do I apply for a loan?") == "en"


# Cyrillic Uzbek input MUST always return uz_latn — Cyrillic UI is removed
def test_detects_uzbek_cyrillic_as_latin_reply_language():
    assert detect_language("Кредит олиш мумкинми?") == "uz_latn"


def test_detects_uzbek_specific_cyrillic_chars():
    """Letters Ў, Қ, Ғ, Ҳ are uniquely Uzbek Cyrillic → uz_latn."""
    assert detect_language("Ҳамкорлар кимлар?") == "uz_latn"


def test_detects_uzbek_cyrillic_no_unique_chars():
    """Strong Uzbek vocabulary without unique chars still → uz_latn."""
    assert detect_language("Кредит олиш") == "uz_latn"


# ── E-IMZO edge cases ────────────────────────────────────────────────────────

def test_detects_e_imzo_short_query():
    """Classic short query that used to loop at language selection."""
    assert detect_language("E-imzo kalit nima?") == "uz_latn"


def test_detects_e_imzo_variant_no_dash():
    assert detect_language("eimzo nima?") == "uz_latn"


def test_detects_e_imzo_only_cyrillic():
    """E-IMZO typed in Cyrillic context → uz_latn."""
    assert detect_language("E-IMZO калит нима?") == "uz_latn"


def test_detects_elektron_imzo():
    assert detect_language("elektron imzo nima?") == "uz_latn"


# ── normalize_supported_lang ─────────────────────────────────────────────────

def test_normalize_cyrl_to_latin():
    """uz_cyrl is no longer a valid UI lang — must be normalized to uz_latn."""
    assert normalize_supported_lang("uz_cyrl") == "uz_latn"


def test_normalize_valid_langs_unchanged():
    for lang in ("ru", "uz_latn", "en"):
        assert normalize_supported_lang(lang) == lang


def test_normalize_unknown_defaults_to_ru():
    assert normalize_supported_lang("fr") == "ru"
    assert normalize_supported_lang(None) == "ru"


# ── platform_link ────────────────────────────────────────────────────────────

def test_platform_link_uz_latn():
    assert platform_link("uz_latn") == "https://finko.uz/uz"


def test_platform_link_ru():
    assert platform_link("ru") == "https://finko.uz/ru"


def test_platform_link_en():
    assert platform_link("en") == "https://finko.uz/ru"


def test_platform_link_cyrl_normalized():
    """uz_cyrl → uz_latn → finko.uz/uz"""
    assert platform_link("uz_cyrl") == "https://finko.uz/uz"


# ── UI keyboards ─────────────────────────────────────────────────────────────

def test_keyboard_fallback_for_unknown_language():
    keyboard = get_keyboard_for_lang("unknown")
    assert keyboard["keyboard"][0][0]["text"] == "Кредиты"


def test_language_keyboard_has_no_uzbek_cyrillic_option():
    keyboard = get_language_keyboard()
    flat_text = " ".join(button["text"] for row in keyboard["keyboard"] for button in row)
    assert "Кирилл" not in flat_text
    assert "O'zbek (Lotin)" in flat_text


def test_language_keyboard_has_three_options():
    keyboard = get_language_keyboard()
    buttons = [btn for row in keyboard["keyboard"] for btn in row]
    assert len(buttons) == 3


def test_get_keyboard_uz_latn():
    keyboard = get_keyboard_for_lang("uz_latn")
    assert keyboard["keyboard"][0][0]["text"] == "Kreditlar"
    assert keyboard["keyboard"][0][1]["text"] == "Biznes"


def test_get_keyboard_en():
    keyboard = get_keyboard_for_lang("en")
    assert keyboard["keyboard"][0][0]["text"] == "Credits"
