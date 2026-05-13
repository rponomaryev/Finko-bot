import re

UZ_CYR_SPECIFIC_RE = re.compile(r"[ЎўҚқҒғҲҳ]")
ANY_CYR_RE = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳ]")
LATIN_RE = re.compile(r"[A-Za-z]")

UZ_LATN_HINTS_RE = re.compile(
    r"\b("
    r"yo'q|ha|salom|assalomu|rahmat|iltimos|bo'yicha|qanday|qanaqa|mumkin|kerak|"
    r"ariza|banklar|hamkorlar|mijoz|foiz|muddat|shartlar|tasdiq|rad|hujjat|"
    r"to'lov|o'zbek|uzbek|qaysi|bilan|ishlaysiz|kompaniya|tashkilot|"
    r"ma'lumot|mavjud|kerakmi|bormi|qiladi|qilinadi|yoki|hamkor|biznes|"
    r"aloqa|kontakt|qarz|mfo|mmt|moliya|savol|javob|foydalanuvchi|"
    r"kreditlar|kontaktlar|hamkorlarga|hamkorlar|olsam|buladi|ish\s*haq|"
    r"kredit\s*ol|qanday\s*ol|menga|bering|qayerda|hisob|grafik|oylik"
    r")\b",
    re.IGNORECASE,
)

# Extended Cyrillic Uzbek strong hints — covers cases without special chars
UZ_CYRL_STRONG_HINTS_RE = re.compile(
    r"\b("
    r"салом|ассалому|рахмат|илтимос|қандай|қайси|мумкин|керак|"
    r"ариза|банклар|ҳамкорлар|мижоз|фоиз|муддат|шартлар|ҳужжат|"
    r"тўлов|маълумот|мавжуд|алоқа|қарз|ммт|жавоб|савол|"
    r"фойдаланувчи|ҳамкорларга|"
    # Without special chars — still clearly Uzbek Cyrillic vocabulary:
    r"менинг|сизнинг|уларнинг|биздан|сиздан|улардан|"
    r"кредит\s*олиш|кредит\s*бериш|кредит\s*олса|кредит\s*бор|"
    r"иш\s*ҳақи|иш\s*хаки|иш\s*хақи|"
    r"нима\s*учун|қачон|қаерда|нарса|ҳамма|барча|"
    r"тасдиқ|рад\s*этиш|ҳисоб|жадвал|ойлик\s*тўлов"
    r")\b",
    re.IGNORECASE,
)

# Additional Cyrillic-only Uzbek word hints (no special chars needed)
UZ_CYRL_VOCAB_RE = re.compile(
    r"\b("
    r"олиш|бериш|берилади|топшириш|топширинг|кирасиз|"
    r"банк|кредит|депозит|суғурта|лизинг|"  # these in Cyrillic context
    r"йўқ|бор|керак|мумкин|яхши|"           # common Uzbek words in Cyrillic
    r"нима|ким|қани|қачон|қанча|"
    r"биринчи|иккинчи|учинчи|"
    r"катта|кичик|кўп|оз"
    r")\b",
    re.IGNORECASE,
)

EN_HINTS_RE = re.compile(
    r"\b("
    r"hello|hi|thanks|please|loan|credit|application|status|bank|banks|"
    r"partner|partners|insurance|leasing|how|what|where|when|can|do|does|"
    r"is|are|which|work|with|company|information|available|customer|"
    r"business|contact|contacts|support|help|mortgage|microfinance|"
    r"calculate|payment|schedule|monthly|annual|rate|amount|term"
    r")\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    lowered = text.lower().strip()

    if not lowered:
        return "unknown"

    # Special Uzbek Cyrillic chars → definitive uz_cyrl
    if UZ_CYR_SPECIFIC_RE.search(text):
        return "uz_cyrl"

    # Any Cyrillic chars present
    if ANY_CYR_RE.search(text):
        if UZ_CYRL_STRONG_HINTS_RE.search(lowered):
            return "uz_cyrl"
        # Extra vocab check: if several Uzbek-only Cyrillic words appear
        cyrl_vocab_matches = len(UZ_CYRL_VOCAB_RE.findall(lowered))
        if cyrl_vocab_matches >= 2:
            return "uz_cyrl"
        return "ru"

    # Latin script: score Uzbek vs English
    uz_score = len(UZ_LATN_HINTS_RE.findall(lowered))
    en_score = len(EN_HINTS_RE.findall(lowered))

    if any(x in lowered for x in ["o'", "g'", "yo'q", "ya'ni", "o'z"]):
        uz_score += 3

    if re.search(
        r"\b(kredit|kreditlar|qarz|kerak|menga|olmoqchiman|bering|bormi|"
        r"qayerda|qanday|olsam|buladi|ish\s*haq|hisob|grafik|oylik)\b",
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
            "application", "apply", "office", "address",
            "calculate", "payment", "schedule", "monthly", "annual",
        }
        common_uz = {
            "salom", "assalomu", "rahmat", "kredit", "kreditlar", "biznes",
            "hamkorlar", "ariza", "aloqa", "kontaktlar", "kerak", "mumkin",
            "ha", "yoq", "yo'q", "yordam", "mijoz", "foiz", "muddat",
            "shartlar", "hujjat", "mfo", "mmt", "mikrozaym", "ipoteka",
            "avtokredit", "olsam", "buladi", "hisob", "grafik", "oylik",
        }

        if words & common_en and not words & common_uz:
            return "en"
        if words & common_uz and not words & common_en:
            return "uz_latn"

        return "unknown"

    return "unknown"


def lang_name(lang: str) -> str:
    return {
        "ru": "Russian",
        "uz_latn": "Uzbek written in Latin script",
        "uz_cyrl": "Uzbek written in Cyrillic script",
        "en": "English",
    }.get(lang, "Russian")


def platform_link(lang: str) -> str:
    """Return the correct platform link for the given UI language."""
    if lang in {"uz_latn", "uz_cyrl"}:
        return "https://finko.uz/uz"
    return "https://finko.uz/ru"


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


def rate_limit_message(lang: str, reason: str = "too_many_requests") -> str:
    messages = {
        "ru": "Слишком много запросов подряд. Подождите немного и отправьте вопрос ещё раз.",
        "uz_latn": "Juda ko'p so'rov yuborildi. Biroz kutib, savolni yana yuboring.",
        "uz_cyrl": "Жуда кўп сўров юборилди. Бироз кутиб, саволни яна юборинг.",
        "en": "Too many requests in a row. Please wait a little and send your question again.",
    }
    return messages.get(lang, messages["ru"])


def too_long_message(lang: str) -> str:
    messages = {
        "ru": "Сообщение слишком длинное. Сократите вопрос и отправьте ещё раз.",
        "uz_latn": "Xabar juda uzun. Savolni qisqartirib, qayta yuboring.",
        "uz_cyrl": "Хабар жуда узун. Саволни қисқартириб, қайта юборинг.",
        "en": "The message is too long. Please shorten it and send it again.",
    }
    return messages.get(lang, messages["ru"])
