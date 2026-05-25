import re

UZ_CYR_SPECIFIC_RE = re.compile(r"[ЎўҚқҒғҲҳ]")
ANY_CYR_RE = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳ]")
LATIN_RE = re.compile(r"[A-Za-z]")

UZ_LATN_HINTS_RE = re.compile(
    r"\b("
    r"yo'q|ha|salom|assalomu|rahmat|iltimos|bo'yicha|qanday|qanaqa|nima|kim|qachon|qayerda|qancha|mumkin|kerak|"
    r"ariza|banklar|hamkorlar|mijoz|foiz|muddat|shartlar|tasdiq|rad|hujjat|"
    r"to'lov|o'zbek|uzbek|qaysi|bilan|ishlaysiz|kompaniya|tashkilot|"
    r"ma'lumot|mavjud|kerakmi|bormi|qiladi|qilinadi|yoki|hamkor|biznes|"
    r"aloqa|kontakt|qarz|mfo|mmt|moliya|savol|javob|foydalanuvchi|"
    r"kreditlar|kontaktlar|hamkorlarga|hamkorlar|olsam|buladi|bo'ladi|ish\s*haq|"
    r"kredit\s*ol|qanday\s*ol|menga|bering|hisob|grafik|oylik|"
    r"imzo|eimzo|kalit|sertifikat|elektron|raqamli|parol"
    r")\b",
    re.IGNORECASE,
)

# Uzbek Cyrillic is no longer offered as an interface language, but we still
# recognize common Uzbek Cyrillic wording and answer in Uzbek Latin.
UZ_CYRL_STRONG_HINTS_RE = re.compile(
    r"\b("
    r"салом|ассалому|рахмат|илтимос|қандай|қайси|мумкин|керак|нима|ким|қачон|қаерда|қанча|"
    r"ариза|банклар|ҳамкорлар|мижоз|фоиз|муддат|шартлар|ҳужжат|"
    r"тўлов|маълумот|мавжуд|алоқа|қарз|ммт|жавоб|савол|"
    r"фойдаланувчи|ҳамкорларга|имзо|калит|сертификат|"
    r"менинг|сизнинг|уларнинг|биздан|сиздан|улардан|"
    r"кредит\s*олиш|кредит\s*бериш|кредит\s*олса|кредит\s*бор|"
    r"иш\s*ҳақи|иш\s*хаки|иш\s*хақи|"
    r"нима\s*учун|нарса|ҳамма|барча|"
    r"тасдиқ|рад\s*этиш|ҳисоб|жадвал|ойлик\s*тўлов"
    r")\b",
    re.IGNORECASE,
)

UZ_CYRL_VOCAB_RE = re.compile(
    r"\b("
    r"олиш|бериш|берилади|топшириш|топширинг|кирасиз|"
    r"кредит|депозит|суғурта|лизинг|йўқ|бор|керак|мумкин|яхши|"
    r"нима|ким|қани|қачон|қанча|биринчи|иккинчи|учинчи|катта|кичик|кўп|оз|имзо|калит"
    r")\b",
    re.IGNORECASE,
)

EN_HINTS_RE = re.compile(
    r"\b("
    r"hello|hi|thanks|please|loan|credit|application|status|bank|banks|"
    r"partner|partners|insurance|leasing|how|what|where|when|can|do|does|"
    r"is|are|which|work|with|company|information|available|customer|"
    r"business|contact|contacts|support|help|mortgage|microfinance|"
    r"calculate|payment|schedule|monthly|annual|rate|amount|term|"
    r"digital|signature|certificate|key"
    r")\b",
    re.IGNORECASE,
)


def normalize_supported_lang(lang: str | None) -> str:
    if lang == "uz_cyrl":
        return "uz_latn"
    if lang in {"ru", "uz_latn", "en"}:
        return lang
    return "ru"


def detect_language(text: str) -> str:
    lowered = text.lower().strip()

    if not lowered:
        return "unknown"

    # E-IMZO questions are commonly written in Uzbek Latin and may contain only
    # short words such as "E-imzo kalit nima?".
    if re.search(r"\be\s*-?\s*imzo\b|\beimzo\b|elektron\s+imzo", lowered):
        if not ANY_CYR_RE.search(text):
            return "uz_latn"

    # Uzbek Cyrillic input is answered in Uzbek Latin because Cyrillic UI was removed.
    if UZ_CYR_SPECIFIC_RE.search(text):
        return "uz_latn"

    if ANY_CYR_RE.search(text):
        if UZ_CYRL_STRONG_HINTS_RE.search(lowered):
            return "uz_latn"
        cyrl_vocab_matches = len(UZ_CYRL_VOCAB_RE.findall(lowered))
        if cyrl_vocab_matches >= 2:
            return "uz_latn"
        return "ru"

    uz_score = len(UZ_LATN_HINTS_RE.findall(lowered))
    en_score = len(EN_HINTS_RE.findall(lowered))

    if any(x in lowered for x in ["o'", "g'", "yo'q", "ya'ni", "o'z"]):
        uz_score += 3

    if re.search(r"\be\s*-?\s*imzo\b|\beimzo\b|elektron\s+imzo", lowered):
        uz_score += 4

    if re.search(
        r"\b(kredit|kreditlar|qarz|kerak|nima|menga|olmoqchiman|bering|bormi|"
        r"qayerda|qanday|olsam|buladi|bo'ladi|ish\s*haq|hisob|grafik|oylik|imzo|kalit)\b",
        lowered,
    ):
        uz_score += 2

    if en_score > uz_score:
        return "en"

    if uz_score > en_score:
        return "uz_latn"

    if LATIN_RE.search(text):
        words = set(re.findall(r"[a-zA-Z']+", lowered))

        common_en = {
            "the", "what", "which", "hello", "hi", "bank", "banks",
            "contacts", "contact", "support", "operator", "status",
            "credit", "loan", "loans", "business", "partners", "help",
            "yes", "no", "ok", "okay", "thanks", "thank", "you",
            "application", "apply", "office", "address", "calculate",
            "payment", "schedule", "monthly", "annual", "digital", "signature",
            "certificate", "key",
        }
        common_uz = {
            "salom", "assalomu", "rahmat", "kredit", "kreditlar", "biznes",
            "hamkorlar", "ariza", "aloqa", "kontaktlar", "kerak", "mumkin",
            "ha", "yoq", "yo'q", "yordam", "mijoz", "foiz", "muddat",
            "shartlar", "hujjat", "mfo", "mmt", "mikrozaym", "ipoteka",
            "avtokredit", "olsam", "buladi", "bo'ladi", "hisob", "grafik",
            "oylik", "nima", "imzo", "eimzo", "kalit", "sertifikat",
        }

        if words & common_en and not words & common_uz:
            return "en"
        if words & common_uz and not words & common_en:
            return "uz_latn"

        return "unknown"

    return "unknown"


def lang_name(lang: str) -> str:
    lang = normalize_supported_lang(lang)
    return {
        "ru": "Russian",
        "uz_latn": "Uzbek Latin",
        "en": "English",
    }.get(lang, "Russian")


def platform_link(lang: str) -> str:
    """Return the correct platform link for the given UI language."""
    lang = normalize_supported_lang(lang)
    if lang == "uz_latn":
        return "https://finko.uz/uz"
    return "https://finko.uz/ru"


def not_found_message(lang: str) -> str:
    lang = normalize_supported_lang(lang)
    messages = {
        "ru": "Я не нашёл точную информацию в базе знаний FINKO.",
        "uz_latn": "FINKO bilimlar bazasida aniq ma'lumot topilmadi.",
        "en": "I could not find exact information in the FINKO knowledge base.",
    }
    return messages.get(lang, messages["ru"])


def server_error_message(lang: str) -> str:
    lang = normalize_supported_lang(lang)
    messages = {
        "ru": "Произошла ошибка на сервере. Попробуйте чуть позже.",
        "uz_latn": "Serverda xatolik yuz berdi. Iltimos, biroz keyinroq urinib ko'ring.",
        "en": "A server error occurred. Please try again a little later.",
    }
    return messages.get(lang, messages["ru"])


def quota_error_message(lang: str) -> str:
    lang = normalize_supported_lang(lang)
    messages = {
        "ru": "OpenAI API временно недоступен: проверьте квоту и billing.",
        "uz_latn": "OpenAI API vaqtincha ishlamayapti: kvota va billingni tekshiring.",
        "en": "The OpenAI API is temporarily unavailable: please check quota and billing.",
    }
    return messages.get(lang, messages["ru"])


def rate_limit_message(lang: str, reason: str = "too_many_requests") -> str:
    lang = normalize_supported_lang(lang)
    messages = {
        "ru": "Слишком много запросов подряд. Подождите немного и отправьте вопрос ещё раз.",
        "uz_latn": "Juda ko'p so'rov yuborildi. Biroz kutib, savolni yana yuboring.",
        "en": "Too many requests in a row. Please wait a little and send your question again.",
    }
    return messages.get(lang, messages["ru"])


def too_long_message(lang: str) -> str:
    lang = normalize_supported_lang(lang)
    messages = {
        "ru": "Сообщение слишком длинное. Сократите вопрос и отправьте ещё раз.",
        "uz_latn": "Xabar juda uzun. Savolni qisqartirib, qayta yuboring.",
        "en": "The message is too long. Please shorten it and send it again.",
    }
    return messages.get(lang, messages["ru"])
