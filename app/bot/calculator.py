import re

from app.bot.language import platform_link
from app.bot import state

_CALC_RE = re.compile(
    r"рассчита|посчита|график\s*платеж|график\s*погашени|выплат|ежемесячн|"
    r"сколько\s*(?:мне\s*)?(?:платить|платеж)|платить\s*в\s*месяц|"
    r"hisob|grafik|oylik\s*to'lov|oylik\s*hisob|"
    r"qancha\s*to'layman|qancha\s*tolov|"
    r"ҳисоб|жадвал|ойлик\s*тўлов|"
    r"қанча\s*тўлайман|"
    r"calculat|payment\s*schedul|monthly\s*payment|how\s*much\s*per\s*month",
    re.IGNORECASE,
)

# Annuity / differentiated type keywords
_ANNUITY_RE = re.compile(
    r"аннуитет|annuitet|annuitetli|аннуитетли|annuity|teng\s*to'lov|тенг\s*тўлов",
    re.IGNORECASE,
)
_DIFF_RE = re.compile(
    r"дифференц|differensial|differencial|differenti|"
    r"kamayib\s*boruvchi|каmayadigan|камайиб\s*борувчи|камаядиган",
    re.IGNORECASE,
)


def detect_loan_calc_request(text: str) -> bool:
    return bool(_CALC_RE.search(text))


def parse_loan_params(text: str) -> dict | None:
    """
    Extract amount (UZS), months, annual_rate (%) from free-form text.
    Returns {"amount": float, "months": int, "rate": float} or None.

    Supports examples:
    - "234000000 сум, 36 месяцев, 23%"
    - "234000000, 36 месяцев, 23%"
    - "Срок 36 месяцев, ставка 21%, сумма 234 000 000 сум, аннуитетная"
    - "100 млн сум на 12 месяцев под 26%"
    - "1.2 млрд сум, 60 месяцев, 21%"
    """
    lowered = text.lower().replace("\xa0", " ").replace("_", " ")

    # Number token:
    # - 127.000.000 / 127,000,000 / 127 000 000
    # - 234000000
    # - 1.2 / 1,2 for cases like "1.2 млрд"
    number_re = r"(?:\d{1,3}(?:[., ]\d{3})+|\d+(?:[.,]\d+)?)"

    def parse_number(raw: str) -> float | None:
        raw = raw.strip()
        if not raw:
            return None

        compact = re.sub(r"\s+", "", raw)

        # 127.000.000 or 127,000,000 -> thousands separators
        if re.match(r"^\d{1,3}([.,]\d{3})+$", compact):
            compact = re.sub(r"[.,]", "", compact)
        # 1.5 or 1,5 -> decimal
        elif re.match(r"^\d+[.,]\d+$", compact):
            compact = compact.replace(",", ".")
        else:
            compact = compact.replace(",", "").replace(".", "")

        try:
            return float(compact)
        except ValueError:
            return None

    # ---- Term in months ----
    months: int | None = None

    m = re.search(
        r"(\d+)\s*(?:месяц(?:ев|а)?|мес\.?|oy(?:ga|lik)?|month(?:s)?|ой(?:га|лик)?)",
        lowered,
    )
    if m:
        months = int(m.group(1))

    if months is None:
        m = re.search(r"(\d+)\s*(?:лет|года?|год|year(?:s)?|йил)", lowered)
        if m:
            months = int(m.group(1)) * 12

    # ---- Annual interest rate ----
    rate: float | None = None

    m = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:%|процент(?:а|ов)?|foiz|фоиз|percent)",
        lowered,
    )
    if m:
        try:
            rate = float(m.group(1).replace(",", "."))
        except ValueError:
            pass

    # ---- Amount ----
    amount: float | None = None
    amount_candidates: list[float] = []

    # Prefer numbers explicitly near currency or amount words.
    currency_patterns = [
        # "сумма 234000000", "размер 135.000.000"
        rf"(?:сумма|размер|кредит(?:а)?|kredit\s*summasi|summa|қарз|кредит\s*суммаси)\s*[:\-]?\s*({number_re})(?:\s*(млн\.?|миллион|миллиард|млрд\.?|mln\.?|million|milliard))?",
        # "234000000 сум", "100 млн сум"
        rf"({number_re})(?:\s*(млн\.?|миллион|миллиард|млрд\.?|mln\.?|million|milliard))?\s*(?:сум|sum|uzs|so'm|сўм)",
    ]

    for pattern in currency_patterns:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            raw_num = match.group(1)
            multiplier_word = ""
            if match.lastindex and match.lastindex >= 2:
                multiplier_word = match.group(2) or ""

            value = parse_number(raw_num)
            if value is None:
                continue

            if re.search(r"млрд|миллиард|milliard", multiplier_word, re.IGNORECASE):
                value *= 1_000_000_000
            elif re.search(r"млн|миллион|million|mln", multiplier_word, re.IGNORECASE):
                value *= 1_000_000

            amount_candidates.append(value)

    # Fallback: collect all numbers, exclude month/rate-sized values, take the largest.
    if not amount_candidates:
        fallback_pattern = rf"({number_re})(?:\s*(млн\.?|миллион|миллиард|млрд\.?|mln\.?|million|milliard))?"
        for match in re.finditer(fallback_pattern, lowered, flags=re.IGNORECASE):
            raw_num = match.group(1)
            multiplier_word = match.group(2) or ""

            value = parse_number(raw_num)
            if value is None:
                continue

            if re.search(r"млрд|миллиард|milliard", multiplier_word, re.IGNORECASE):
                value *= 1_000_000_000
            elif re.search(r"млн|миллион|million|mln", multiplier_word, re.IGNORECASE):
                value *= 1_000_000

            # Filter out likely month/rate values such as 12, 21, 36, 60.
            if value >= 1_000:
                amount_candidates.append(value)

    if amount_candidates:
        amount = max(amount_candidates)

    if amount and months and rate:
        return {"amount": amount, "months": months, "rate": rate}

    return None


def detect_payment_type(text: str) -> str | None:
    if _ANNUITY_RE.search(text):
        return "annuity"
    if _DIFF_RE.search(text):
        return "differentiated"
    return None


def _fmt(n: float) -> str:
    """Format integer with space thousands separator."""
    return f"{round(n):,}".replace(",", " ")


def calc_annuity(amount: float, rate_annual: float, months: int) -> list[dict]:
    i = rate_annual / 12 / 100
    if i == 0:
        payment = amount / months
    else:
        factor = (1 + i) ** months
        payment = amount * i * factor / (factor - 1)

    schedule = []
    balance = amount
    for mo in range(1, months + 1):
        interest = balance * i
        principal = payment - interest
        balance = max(balance - principal, 0.0)
        schedule.append({
            "month": mo,
            "payment": round(payment),
            "interest": round(interest),
            "principal": round(principal),
            "balance": round(balance),
        })
    return schedule


def calc_differentiated(amount: float, rate_annual: float, months: int) -> list[dict]:
    i = rate_annual / 12 / 100
    principal_per_month = amount / months
    schedule = []
    balance = amount
    for mo in range(1, months + 1):
        interest = balance * i
        payment = principal_per_month + interest
        balance = max(balance - principal_per_month, 0.0)
        schedule.append({
            "month": mo,
            "payment": round(payment),
            "interest": round(interest),
            "principal": round(principal_per_month),
            "balance": round(balance),
        })
    return schedule


def format_loan_schedule(
    schedule: list[dict],
    payment_type: str,
    lang: str,
    amount: float,
    rate: float,
    months: int,
) -> str:
    total_payment = sum(r["payment"] for r in schedule)
    total_interest = sum(r["interest"] for r in schedule)
    link = platform_link(lang)

    first_payment = schedule[0]["payment"]
    last_payment = schedule[-1]["payment"]

    if lang == "ru":
        if payment_type == "annuity":
            return (
                f"📊 Аннуитетный расчёт\n\n"
                f"Сумма кредита: {_fmt(amount)} сум\n"
                f"Срок: {months} месяцев\n"
                f"Ставка: {rate}% годовых\n\n"
                f"Ежемесячный платёж: примерно {_fmt(first_payment)} сум\n"
                f"Итого выплат: {_fmt(total_payment)} сум\n"
                f"Переплата по процентам: {_fmt(total_interest)} сум\n\n"
                f"⚠️ Это ориентировочный расчёт. Точные условия, комиссии и итоговый график "
                f"определяются банком или МФО.\n\n"
                f"Подать заявку: {link}"
            )

        return (
            f"📊 Дифференцированный расчёт\n\n"
            f"Сумма кредита: {_fmt(amount)} сум\n"
            f"Срок: {months} месяцев\n"
            f"Ставка: {rate}% годовых\n\n"
            f"Первый платёж: примерно {_fmt(first_payment)} сум\n"
            f"Последний платёж: примерно {_fmt(last_payment)} сум\n"
            f"Итого выплат: {_fmt(total_payment)} сум\n"
            f"Переплата по процентам: {_fmt(total_interest)} сум\n\n"
            f"⚠️ Это ориентировочный расчёт. Точные условия, комиссии и итоговый график "
            f"определяются банком или МФО.\n\n"
            f"Подать заявку: {link}"
        )

    if lang == "uz_latn":
        if payment_type == "annuity":
            return (
                f"📊 Annuitet hisob-kitob\n\n"
                f"Kredit summasi: {_fmt(amount)} so'm\n"
                f"Muddat: {months} oy\n"
                f"Stavka: {rate}% yillik\n\n"
                f"Oylik to'lov: taxminan {_fmt(first_payment)} so'm\n"
                f"Jami to'lov: {_fmt(total_payment)} so'm\n"
                f"Foiz bo'yicha ortiqcha to'lov: {_fmt(total_interest)} so'm\n\n"
                f"⚠️ Bu taxminiy hisob-kitob. Aniq shartlar, komissiyalar va yakuniy grafik "
                f"bank yoki MMT tomonidan belgilanadi.\n\n"
                f"Ariza topshirish: {link}"
            )

        return (
            f"📊 Differensial hisob-kitob\n\n"
            f"Kredit summasi: {_fmt(amount)} so'm\n"
            f"Muddat: {months} oy\n"
            f"Stavka: {rate}% yillik\n\n"
            f"Birinchi to'lov: taxminan {_fmt(first_payment)} so'm\n"
            f"Oxirgi to'lov: taxminan {_fmt(last_payment)} so'm\n"
            f"Jami to'lov: {_fmt(total_payment)} so'm\n"
            f"Foiz bo'yicha ortiqcha to'lov: {_fmt(total_interest)} so'm\n\n"
            f"⚠️ Bu taxminiy hisob-kitob. Aniq shartlar, komissiyalar va yakuniy grafik "
            f"bank yoki MMT tomonidan belgilanadi.\n\n"
            f"Ariza topshirish: {link}"
        )

    if lang == "uz_cyrl":
        if payment_type == "annuity":
            return (
                f"📊 Аннуитет ҳисоб-китоб\n\n"
                f"Кредит суммаси: {_fmt(amount)} сўм\n"
                f"Муддат: {months} ой\n"
                f"Ставка: {rate}% йиллик\n\n"
                f"Ойлик тўлов: тахминан {_fmt(first_payment)} сўм\n"
                f"Жами тўлов: {_fmt(total_payment)} сўм\n"
                f"Фоиз бўйича ортиқча тўлов: {_fmt(total_interest)} сўм\n\n"
                f"⚠️ Бу тахминий ҳисоб-китоб. Аниқ шартлар, комиссиялар ва якуний график "
                f"банк ёки ММТ томонидан белгиланади.\n\n"
                f"Ариза топшириш: {link}"
            )

        return (
            f"📊 Дифференциал ҳисоб-китоб\n\n"
            f"Кредит суммаси: {_fmt(amount)} сўм\n"
            f"Муддат: {months} ой\n"
            f"Ставка: {rate}% йиллик\n\n"
            f"Биринчи тўлов: тахминан {_fmt(first_payment)} сўм\n"
            f"Охирги тўлов: тахминан {_fmt(last_payment)} сўм\n"
            f"Жами тўлов: {_fmt(total_payment)} сўм\n"
            f"Фоиз бўйича ортиқча тўлов: {_fmt(total_interest)} сўм\n\n"
            f"⚠️ Бу тахминий ҳисоб-китоб. Аниқ шартлар, комиссиялар ва якуний график "
            f"банк ёки ММТ томонидан белгиланади.\n\n"
            f"Ариза топшириш: {link}"
        )

    if payment_type == "annuity":
        return (
            f"📊 Annuity calculation\n\n"
            f"Loan amount: {_fmt(amount)} UZS\n"
            f"Term: {months} months\n"
            f"Rate: {rate}% per year\n\n"
            f"Monthly payment: about {_fmt(first_payment)} UZS\n"
            f"Total payments: {_fmt(total_payment)} UZS\n"
            f"Interest overpayment: {_fmt(total_interest)} UZS\n\n"
            f"⚠️ This is an estimated calculation. Actual terms, fees, and the final schedule "
            f"are set by the bank or MFO.\n\n"
            f"Apply: {link}"
        )

    return (
        f"📊 Differentiated calculation\n\n"
        f"Loan amount: {_fmt(amount)} UZS\n"
        f"Term: {months} months\n"
        f"Rate: {rate}% per year\n\n"
        f"First payment: about {_fmt(first_payment)} UZS\n"
        f"Last payment: about {_fmt(last_payment)} UZS\n"
        f"Total payments: {_fmt(total_payment)} UZS\n"
        f"Interest overpayment: {_fmt(total_interest)} UZS\n\n"
        f"⚠️ This is an estimated calculation. Actual terms, fees, and the final schedule "
        f"are set by the bank or MFO.\n\n"
        f"Apply: {link}"
    )


def ask_payment_type_message(lang: str) -> str:
    messages = {
        "ru": (
            "Какой тип платежа рассчитать?\n\n"
            "1️⃣ Аннуитетный — одинаковый платёж каждый месяц.\n"
            "2️⃣ Дифференцированный — платёж постепенно уменьшается, "
            "основной долг гасится равными частями.\n\n"
            "Напишите «аннуитетный» или «дифференцированный»."
        ),
        "uz_latn": (
            "Qaysi to'lov turini hisoblash kerak?\n\n"
            "1️⃣ Annuitet — har oy bir xil to'lov.\n"
            "2️⃣ Differensial — to'lov asta-sekin kamayib boradi, "
            "asosiy qarz teng qismlarda to'lanadi.\n\n"
            "«Annuitet» yoki «differensial» deb yozing."
        ),
        "uz_cyrl": (
            "Қайси тўлов турини ҳисоблаш керак?\n\n"
            "1️⃣ Аннуитет — ҳар ой бир хил тўлов.\n"
            "2️⃣ Дифференциал — тўлов аста-секин камайиб боради, "
            "асосий қарз тенг қисмларда тўланади.\n\n"
            "«Аннуитет» ёки «дифференциал» деб ёзинг."
        ),
        "en": (
            "Which payment type should I calculate?\n\n"
            "1️⃣ Annuity — equal payment every month.\n"
            "2️⃣ Differentiated — payment gradually decreases as "
            "the principal is repaid in equal parts.\n\n"
            "Reply with 'annuity' or 'differentiated'."
        ),
    }
    return messages.get(lang, messages["ru"])


async def handle_loan_calc(chat_id: int, user_text: str, lang: str) -> str | None:
    pending = await state.get_pending_calc(chat_id)
    params = parse_loan_params(user_text)
    payment_type = detect_payment_type(user_text)

    # 1. User first selected payment type, then sent loan parameters.
    if pending and params and pending.get("payment_type"):
        saved_payment_type = pending["payment_type"]
        await state.clear_pending_calc(chat_id)

        if saved_payment_type == "annuity":
            schedule = calc_annuity(params["amount"], params["rate"], params["months"])
        else:
            schedule = calc_differentiated(params["amount"], params["rate"], params["months"])

        return format_loan_schedule(
            schedule,
            saved_payment_type,
            lang,
            params["amount"],
            params["rate"],
            params["months"],
        )

    # 2. We already have loan parameters and are waiting for payment type.
    if pending and payment_type and "amount" in pending and "rate" in pending and "months" in pending:
        await state.clear_pending_calc(chat_id)

        if payment_type == "annuity":
            schedule = calc_annuity(pending["amount"], pending["rate"], pending["months"])
        else:
            schedule = calc_differentiated(pending["amount"], pending["rate"], pending["months"])

        return format_loan_schedule(
            schedule,
            payment_type,
            lang,
            pending["amount"],
            pending["rate"],
            pending["months"],
        )

    # 3. User sent only payment type without loan parameters.
    if payment_type and not params:
        await state.set_pending_calc(chat_id, {"payment_type": payment_type})
        messages = {
            "ru": "Хорошо. Теперь укажите сумму кредита, срок и процентную ставку. Например: 234000000 сум, 36 месяцев, 23%.",
            "uz_latn": "Yaxshi. Endi kredit summasi, muddati va foiz stavkasini yozing. Masalan: 234000000 so'm, 36 oy, 23%.",
            "uz_cyrl": "Яхши. Энди кредит суммаси, муддати ва фоиз ставкаcини ёзинг. Масалан: 234000000 сўм, 36 ой, 23%.",
            "en": "Okay. Now send the loan amount, term, and interest rate. Example: 234000000 UZS, 36 months, 23%.",
        }
        return messages.get(lang, messages["ru"])

    # 4. User sent parameters and payment type in one message.
    if params and payment_type:
        if payment_type == "annuity":
            schedule = calc_annuity(params["amount"], params["rate"], params["months"])
        else:
            schedule = calc_differentiated(params["amount"], params["rate"], params["months"])

        return format_loan_schedule(
            schedule,
            payment_type,
            lang,
            params["amount"],
            params["rate"],
            params["months"],
        )

    # 5. User sent parameters but did not specify payment type.
    if params:
        await state.set_pending_calc(chat_id, params)
        return ask_payment_type_message(lang)

    # 6. User asks for a calculation but did not provide enough data.
    if detect_loan_calc_request(user_text):
        messages = {
            "ru": "Укажите, пожалуйста, сумму кредита, срок и процентную ставку. Например: 234000000 сум, 36 месяцев, 23%.",
            "uz_latn": "Iltimos, kredit summasi, muddati va foiz stavkasini yozing. Masalan: 234000000 so'm, 36 oy, 23%.",
            "uz_cyrl": "Илтимос, кредит суммаси, муддати ва фоиз ставкаcини ёзинг. Масалан: 234000000 сўм, 36 ой, 23%.",
            "en": "Please provide the loan amount, term, and interest rate. Example: 234000000 UZS, 36 months, 23%.",
        }
        return messages.get(lang, messages["ru"])

    return None
