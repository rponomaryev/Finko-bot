import pytest

from app.bot import state
from app.bot.calculator import calc_annuity, handle_loan_calc, parse_loan_params


def test_parse_loan_params():
    params = parse_loan_params("234000000 сум, 36 месяцев, 23%")
    assert params == {"amount": 234_000_000.0, "months": 36, "rate": 23.0}


def test_annuity_schedule_len_and_equal_payments():
    schedule = calc_annuity(234_000_000, 23, 36)
    assert len(schedule) == 36
    assert schedule[0]["payment"] == schedule[-1]["payment"]
    assert schedule[0]["payment"] > 0


@pytest.mark.asyncio
async def test_pending_calculator_state_flow():
    chat_id = 99001
    await state.clear_chat_state(chat_id)

    first = await handle_loan_calc(chat_id, "234000000 сум, 36 месяцев, 23%", "ru")
    assert "тип платежа" in first.lower()

    pending = await state.get_pending_calc(chat_id)
    assert pending["amount"] == 234_000_000.0

    second = await handle_loan_calc(chat_id, "аннуитетный", "ru")
    assert "Аннуитетный расчёт" in second
    assert await state.get_pending_calc(chat_id) is None
