from app.bot.intents import detect_intent, normalize_text_for_match
from app.bot.language import normalize_supported_lang, platform_link
from app.bot import state


def _partner_text(lang: str) -> str:
    if lang == "uz_latn":
        return (
            "FINKO Hamkorbank, Universal Bank, DavrBank, Madad Invest Bank va Garant Bank bilan hamkorlik qiladi. "
            "MFO/MMT hamkorlari: DELTA, PULMAN, APEX MOLIYA, ANSOR, ALMIZAN, "
            "Aloqa Miqromoliya Tashkiloti, Una Moliya va VAFO MOLIYA. "
            "Hamkor tashkilotlar soni doimiy ravishda oshib bormoqda."
        )
    if lang == "en":
        return (
            "FINKO works with Hamkorbank, Universal Bank, DavrBank, Madad Invest Bank, and Garant Bank. "
            "Partner MFOs are DELTA, PULMAN, APEX MOLIYA, ANSOR, ALMIZAN, "
            "Aloqa Miqromoliya Tashkiloti, Una Moliya, and VAFO MOLIYA. "
            "The number of partner organizations is continuously growing."
        )
    return (
        "FINKO сотрудничает с Hamkorbank, Universal Bank, DavrBank, Madad Invest Bank и Garant Bank. "
        "МФО-партнёры: DELTA, PULMAN, APEX MOLIYA, ANSOR, ALMIZAN, "
        "Aloqa Miqromoliya Tashkiloti, Una Moliya и VAFO MOLIYA. "
        "Количество партнёрских организаций постоянно увеличивается."
    )


def _e_imzo_text(lang: str) -> str:
    if lang == "uz_latn":
        return (
            "E-IMZO kaliti — bu O'zbekistonda elektron raqamli imzo uchun ishlatiladigan shaxsiy kalit/sertifikat. "
            "U shaxsni tasdiqlash va hujjatlarni onlayn imzolash uchun kerak bo'ladi. "
            "FINKO'da E-IMZO hamkor tashkilot bilan hujjatlarni imzolash bosqichida talab qilinishi mumkin. "
            "Kalit va parolni uchinchi shaxslarga bermang."
        )
    if lang == "en":
        return (
            "E-IMZO is an electronic digital signature key/certificate used in Uzbekistan. "
            "It confirms a person's identity and allows documents to be signed online. "
            "In FINKO, E-IMZO may be required when signing documents with a partner organization. "
            "Do not share your key or password with anyone."
        )
    return (
        "E-IMZO — это ключ/сертификат электронной цифровой подписи в Узбекистане. "
        "Он подтверждает личность и позволяет подписывать документы онлайн. "
        "В FINKO E-IMZO может понадобиться на этапе подписания документов с партнёрской организацией. "
        "Не передавайте ключ и пароль третьим лицам."
    )


def build_quick_answer(action: str, lang: str) -> str:
    lang = normalize_supported_lang(lang)
    link = platform_link(lang)
    answers = {
        "restart": {
            "ru": "Бот перезапущен. Можете отправить новый вопрос.",
            "uz_latn": "Bot qayta ishga tushdi. Yangi savol yuborishingiz mumkin.",
            "en": "The bot has been restarted. You can send a new question.",
        },
        "insert_question": {
            "ru": "Напишите ваш вопрос одним сообщением. Я определю язык вопроса и отвечу на этом языке.",
            "uz_latn": "Savolingizni bitta xabarda yozing. Men savol tilini aniqlab, shu tilda javob beraman.",
            "en": "Please write your question in one message. I will detect the language and reply in that language.",
        },
        "contacts": {
            "ru": (
                "Контакты FINKO:\n"
                "Телефон: +998 50 177 77 88\n"
                "Email: info@finko.uz\n"
                "Сайт: https://finko.uz\n"
                "Офис: Ташкент, ул. Ойбек 18/1, БЦ ATRIUM\n"
                "Telegram: https://t.me/finkouz\n"
                "Рабочее время: с понедельника по пятницу, с 9.00 до 18.00"
            ),
            "uz_latn": (
                "FINKO kontaktlari:\n"
                "Telefon: +998 50 177 77 88\n"
                "Email: info@finko.uz\n"
                "Sayt: https://finko.uz\n"
                "Ofis: Toshkent, Oybek 18/1, ATRIUM\n"
                "Telegram: https://t.me/finkouz\n"
                "Ish vaqti: dushanbadan jumagacha, soat 9:00 dan 18:00 gacha"
            ),
            "en": (
                "FINKO contacts:\n"
                "Phone: +998 50 177 77 88\n"
                "Email: info@finko.uz\n"
                "Website: https://finko.uz\n"
                "Office: Tashkent, Oybek 18/1, ATRIUM\n"
                "Telegram: https://t.me/finkouz\n"
                "Working hours: Monday to Friday, from 9:00 to 18:00"
            ),
        },
        "credits_menu": {
            "ru": (
                f"Через FINKO доступны потребительские кредиты, автокредиты, ипотека, "
                f"микрозаймы и другие финансовые продукты. Условия по сумме, сроку и ставке "
                f"определяются банком или МФО-партнёром. FINKO не выдаёт кредиты напрямую.\n\n"
                f"Подать заявку: {link}"
            ),
            "uz_latn": (
                f"FINKO orqali iste'mol kreditlari, avtokreditlar, ipoteka, mikrozaymlar "
                f"va boshqa moliyaviy mahsulotlar mavjud. Summa, muddat va stavka hamkor bank "
                f"yoki MMT tomonidan belgilanadi. FINKO kreditni to'g'ridan-to'g'ri bermaydi.\n\n"
                f"Ariza topshirish: {link}"
            ),
            "en": (
                f"Through FINKO, users can access consumer loans, auto loans, mortgages, "
                f"microloans, and other financial products. The amount, term, and rate are "
                f"set by the partner bank or MFO. FINKO does not issue loans directly.\n\n"
                f"Apply now: {link}"
            ),
        },
        "business_menu": {
            "ru": (
                f"Для бизнеса через FINKO доступны бизнес-кредиты, оборотные и инвестиционные "
                f"кредиты, лизинг, вклады, страхование и другие решения. "
                f"Итоговые условия определяются партнёрской организацией.\n\n"
                f"Подать заявку: {link}"
            ),
            "uz_latn": (
                f"Biznes uchun FINKO orqali biznes kreditlari, aylanma va investitsiya "
                f"kreditlari, lizing, depozitlar, sug'urta va boshqa yechimlar mavjud. "
                f"Yakuniy shartlar hamkor tashkilot tomonidan belgilanadi.\n\n"
                f"Ariza topshirish: {link}"
            ),
            "en": (
                f"For businesses, FINKO offers access to business loans, working capital and "
                f"investment loans, leasing, deposits, insurance, and related solutions. "
                f"Final terms are set by the partner organization.\n\n"
                f"Apply now: {link}"
            ),
        },
        "partners_menu": {
            "ru": _partner_text("ru"),
            "uz_latn": _partner_text("uz_latn"),
            "en": _partner_text("en"),
        },
        "e_imzo": {
            "ru": _e_imzo_text("ru"),
            "uz_latn": _e_imzo_text("uz_latn"),
            "en": _e_imzo_text("en"),
        },
        "greeting": {
            "ru": "Здравствуйте! Я AI-ассистент FINKO. Могу помочь с продуктами, бизнес-вопросами, партнёрством и контактами.",
            "uz_latn": "Salom! Men FINKO AI yordamchisiman. Mahsulotlar, biznes savollari, hamkorlik va kontaktlar bo'yicha yordam bera olaman.",
            "en": "Hello! I'm the FINKO AI assistant. I can help with products, business questions, partnerships, and contacts.",
        },
        "thanks": {
            "ru": "Пожалуйста! Если захотите, можете задать ещё один вопрос.",
            "uz_latn": "Marhamat! Xohlasangiz, yana savol yuborishingiz mumkin.",
            "en": "You're welcome! Feel free to send another question.",
        },
    }

    if action not in answers:
        return ""

    return answers[action].get(lang, answers[action]["ru"])


def should_use_quick_reply(intent: str, user_text: str) -> bool:
    text = normalize_text_for_match(user_text)

    if intent in {
        "restart", "insert_question", "contacts", "credits_menu", "business_menu",
        "partners_menu", "e_imzo", "greeting", "thanks"
    }:
        return True

    if len(text.split()) <= 3 and intent in {"credits", "business", "partners"}:
        return True

    return False


async def handle_menu_or_quick_action(
    user_text: str,
    chat_id: int,
    ui_lang: str,
) -> tuple[str | None, str | None]:
    intent = detect_intent(user_text)

    if not should_use_quick_reply(intent, user_text):
        return None, intent

    if intent == "restart":
        await state.clear_chat_state(chat_id)

    quick_intent = intent
    if intent == "credits":
        quick_intent = "credits_menu"
    elif intent == "business":
        quick_intent = "business_menu"
    elif intent == "partners":
        quick_intent = "partners_menu"

    answer = build_quick_answer(quick_intent, ui_lang)
    return answer or None, intent
