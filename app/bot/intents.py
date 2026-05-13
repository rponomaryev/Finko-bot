import re

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
        "/question", "/ask", "ask a question", "insert my question", "write a question",
        "задать вопрос", "ввести вопрос", "написать вопрос", "свой вопрос",
        "savol yozish", "savol berish", "savolim bor", "savol yuborish",
        "савол ёзиш", "савол бериш", "саволим бор", "савол юбориш"
    }:
        return "insert_question"

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

    if text in {"бизнес", "biznes", "business", "/business"}:
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

    # Loan calculation intent — checked before generic keywords
    calc_keywords_ru = ["рассчита", "посчита", "график платеж", "выплат", "ежемесячный платёж", "ежемесячный платеж"]
    calc_keywords_uz = ["hisob", "grafik", "oylik to'lov", "ҳисоб", "жадвал", "ойлик тўлов"]
    calc_keywords_en = ["calculat", "payment schedule", "monthly payment", "amortiz"]

    for kw in calc_keywords_ru + calc_keywords_uz + calc_keywords_en:
        if kw in text:
            return "loan_calc"

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
