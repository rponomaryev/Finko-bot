from typing import Any

def extract_user_message(update: dict[str, Any]) -> tuple[int | None, str | None, dict[str, Any]]:
    message = update.get("message")
    if not message:
        return None, None, {}

    chat = message.get("chat", {})
    from_user = message.get("from", {})

    chat_id = chat.get("id")
    text = message.get("text")

    if not chat_id or not text:
        return None, None, {}

    profile = {
        "username": from_user.get("username"),
        "first_name": from_user.get("first_name"),
        "last_name": from_user.get("last_name"),
    }

    return chat_id, text.strip(), profile
