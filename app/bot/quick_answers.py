from app.bot.intents import detect_intent, normalize_text_for_match
from app.bot.language import platform_link
from app.bot import state

def build_quick_answer(action: str, lang: str) -> str:
    link = platform_link(lang)
    answers = {
        "restart": {
            "ru": "Бот перезапущен. Можете отправить новый вопрос.",
            "uz_latn": "Bot qayta ishga tushdi. Yangi savol yuborishingiz mumkin.",
            "uz_cyrl": "Бот қайта ишга тушди. Янги савол юборишингиз мумкин.",
            "en": "The bot has been restarted. You can send a new question.",
        },
        "insert_question": {
            "ru": "Напишите ваш вопрос одним сообщением. Я определю язык вопроса и отвечу на этом языке.",
            "uz_latn": "Savolingizni bitta xabarda yozing. Men savol tilini aniqlab, shu tilda javob beraman.",
            "uz_cyrl": "Саволингизни битта хабарда ёзинг. Мен савол тилини аниқлаб, шу тилда жавоб бераман.",
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
            "uz_cyrl": (
                "FINKO контактлари:\n"
                "Телефон: +998 50 177 77 88\n"
                "Email: info@finko.uz\n"
                "Сайт: https://finko.uz\n"
                "Офис: Тошкент, Ойбек 18/1, ATRIUM\n"
                "Telegram: https://t.me/finkouz\n"
                "Иш вақти: душанбадан жумагача, соат 9:00 дан 18:00 гача"
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
            "uz_cyrl": (
                f"FINKO орқали истеъмол кредитлари, автокредитлар, ипотека, микрозаймлар "
                f"ва бошқа молиявий маҳсулотлар мавжуд. Сумма, муддат ва ставка ҳамкор банк "
                f"ёки ММТ томонидан белгиланади. FINKO кредитни тўғридан-тўғри бермайди.\n\n"
                f"Ариза топшириш: {link}"
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
            "uz_cyrl": (
                f"Бизнес учун FINKO орқали бизнес кредитлари, айланма ва инвестиция "
                f"кредитлари, лизинг, депозитлар, суғурта ва бошқа ечимлар мавжуд. "
                f"Якуний шартлар ҳамкор ташкилот томонидан белгиланади.\n\n"
                f"Ариза топшириш: {link}"
            ),
            "en": (
                f"For businesses, FINKO offers access to business loans, working capital and "
                f"investment loans, leasing, deposits, insurance, and related solutions. "
                f"Final terms are set by the partner organization.\n\n"
                f"Apply now: {link}"
            ),
        },
        "partners_menu": {
            "ru": (
                "FINKO сотрудничает с Hamkorbank, Universal Bank, DavrBank, Madad Invest Bank, "
                "Garant Bank, Tenge Bank, Asia Alliance Bank, а также с МФО DELTA, PULMAN, "
                "APEX MOLIYA, ANSOR и ALMIZAN. Количество партнёрских "
                "организаций постоянно увеличивается."
            ),
            "uz_latn": (
                "FINKO Hamkorbank, Universal Bank, DavrBank, Madad Invest Bank, Garant Bank, "
                "Tenge Bank, Asia Alliance Bank bilan, shuningdek DELTA, PULMAN, "
                "APEX MOLIYA, ANSOR va ALMIZAN kabi MMTlar bilan "
                "hamkorlik qiladi. Hamkor tashkilotlar soni doimiy ravishda oshib bormoqda."
            ),
            "uz_cyrl": (
                "FINKO Hamkorbank, Universal Bank, DavrBank, Madad Invest Bank, Garant Bank, "
                "Tenge Bank, Asia Alliance Bank билан, шунингдек DELTA, PULMAN, "
                "APEX MOLIYA, ANSOR ва ALMIZAN каби ММТлар билан "
                "ҳамкорлик қилади. Ҳамкор ташкилотлар сони доимий равишда ошиб бормоқда."
            ),
            "en": (
                "FINKO works with Hamkorbank, Universal Bank, DavrBank, Madad Invest Bank, "
                "Garant Bank, Tenge Bank, Asia Alliance Bank, as well as the MFOs DELTA, "
                "PULMAN, APEX MOLIYA, ANSOR, and ALMIZAN. The number "
                "of partner organizations is continuously growing."
            ),
        },
        "greeting": {
            "ru": "Здравствуйте! Я AI-ассистент FINKO. Могу помочь с продуктами, бизнес-вопросами, партнёрством и контактами.",
            "uz_latn": "Salom! Men FINKO AI yordamchisiman. Mahsulotlar, biznes savollari, hamkorlik va kontaktlar bo'yicha yordam bera olaman.",
            "uz_cyrl": "Салом! Мен FINKO AI ёрдамчисиман. Маҳсулотлар, бизнес саволлари, ҳамкорлик ва контактлар бўйича ёрдам бера оламан.",
            "en": "Hello! I'm the FINKO AI assistant. I can help with products, business questions, partnerships, and contacts.",
        },
        "thanks": {
            "ru": "Пожалуйста! Если захотите, можете задать ещё один вопрос.",
            "uz_latn": "Marhamat! Xohlasangiz, yana savol yuborishingiz mumkin.",
            "uz_cyrl": "Марҳамат! Хоҳласангиз, яна савол юборишингиз мумкин.",
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
        "partners_menu", "greeting", "thanks"
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
