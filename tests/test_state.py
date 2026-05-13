import pytest

from app.bot import state
from app.config import CHAT_MEMORY_TURNS


@pytest.mark.asyncio
async def test_memory_respects_configured_turn_count():
    chat_id = 88001
    await state.clear_chat_state(chat_id)
    for i in range(CHAT_MEMORY_TURNS + 3):
        await state.save_assistant_message(chat_id, f"answer-{i}")
    history = await state.get_assistant_history(chat_id, limit=CHAT_MEMORY_TURNS + 10)
    assert len(history) == CHAT_MEMORY_TURNS
    assert history[-1] == f"answer-{CHAT_MEMORY_TURNS + 2}"


@pytest.mark.asyncio
async def test_duplicate_update_detection():
    assert await state.remember_update_id(123456789) is True
    assert await state.remember_update_id(123456789) is False
