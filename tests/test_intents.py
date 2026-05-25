"""Tests for intent detection and quick-answer wiring."""
import pytest
from app.bot.intents import detect_intent
from app.bot.quick_answers import build_quick_answer, should_use_quick_reply


# ── detect_intent ────────────────────────────────────────────────────────────

class TestEImzoIntent:
    def test_e_imzo_dash(self):
        assert detect_intent("E-imzo kalit nima?") == "e_imzo"

    def test_e_imzo_no_dash(self):
        assert detect_intent("eimzo nima bu?") == "e_imzo"

    def test_e_imzo_russian(self):
        assert detect_intent("что такое электронная подпись?") == "e_imzo"

    def test_e_imzo_english(self):
        assert detect_intent("what is a digital signature?") == "e_imzo"

    def test_imzo_kalit(self):
        assert detect_intent("imzo kalit") == "e_imzo"


class TestPartnersIntent:
    def test_partners_ru(self):
        assert detect_intent("партнёры") == "partners_menu"

    def test_partners_uz(self):
        assert detect_intent("hamkorlar") == "partners_menu"

    def test_partners_en(self):
        assert detect_intent("partners") == "partners_menu"

    def test_mfo_keyword(self):
        """MFO keyword → partner intent, not general."""
        assert detect_intent("qanday MFOlar bor?") == "partners"


class TestGreetingIntent:
    def test_salom(self):
        assert detect_intent("salom") == "greeting"

    def test_hello(self):
        assert detect_intent("hello") == "greeting"

    def test_privet(self):
        assert detect_intent("привет") == "greeting"


class TestContactsIntent:
    def test_contacts_en(self):
        assert detect_intent("contacts") == "contacts"

    def test_kontaktlar(self):
        assert detect_intent("kontaktlar") == "contacts"


class TestLoanCalcIntent:
    def test_calculate_ru(self):
        assert detect_intent("рассчитай кредит") == "loan_calc"

    def test_calculate_uz(self):
        assert detect_intent("hisob grafik") == "loan_calc"

    def test_calculate_en(self):
        assert detect_intent("calculate monthly payment") == "loan_calc"


# ── build_quick_answer ───────────────────────────────────────────────────────

class TestQuickAnswers:
    def test_e_imzo_uz_latn_not_empty(self):
        answer = build_quick_answer("e_imzo", "uz_latn")
        assert answer
        assert "E-IMZO" in answer

    def test_e_imzo_ru_not_empty(self):
        answer = build_quick_answer("e_imzo", "ru")
        assert answer
        assert "E-IMZO" in answer

    def test_e_imzo_en_not_empty(self):
        answer = build_quick_answer("e_imzo", "en")
        assert answer
        assert "E-IMZO" in answer

    def test_partners_includes_new_mfos(self):
        for lang in ("ru", "uz_latn", "en"):
            answer = build_quick_answer("partners_menu", lang)
            assert "Aloqa Miqromoliya Tashkiloti" in answer, f"Missing in {lang}"
            assert "Una Moliya" in answer, f"Missing in {lang}"
            assert "VAFO MOLIYA" in answer, f"Missing in {lang}"

    def test_partners_excludes_removed_banks(self):
        for lang in ("ru", "uz_latn", "en"):
            answer = build_quick_answer("partners_menu", lang)
            assert "Tenge" not in answer, f"TengeBank still present in {lang}"
            assert "Asia Alliance" not in answer, f"Asia Alliance still present in {lang}"

    def test_unknown_action_returns_empty(self):
        assert build_quick_answer("nonexistent_action", "ru") == ""

    def test_cyrl_lang_normalized_to_uz_latn(self):
        """uz_cyrl is not a supported lang — must be normalized to uz_latn."""
        answer_cyrl = build_quick_answer("greeting", "uz_cyrl")
        answer_latin = build_quick_answer("greeting", "uz_latn")
        assert answer_cyrl == answer_latin


# ── should_use_quick_reply ───────────────────────────────────────────────────

class TestShouldUseQuickReply:
    def test_e_imzo_always_quick(self):
        assert should_use_quick_reply("e_imzo", "E-imzo kalit nima?") is True

    def test_greeting_always_quick(self):
        assert should_use_quick_reply("greeting", "salom") is True

    def test_general_not_quick(self):
        assert should_use_quick_reply("general", "Kredit qanday ishlaydi?") is False
