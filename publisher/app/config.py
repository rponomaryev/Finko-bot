from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    session_string: str
    publish_key: str
    destination: str = "@finkouz"
    review_destination: int = -5547276399

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_id=int(os.environ["TELEGRAM_API_ID"]),
            api_hash=os.environ["TELEGRAM_API_HASH"],
            session_string=os.environ["TELEGRAM_SESSION_STRING"],
            publish_key=os.environ["FINKO_PUBLISH_KEY"],
            destination=os.environ.get("TELEGRAM_CHANNEL_DESTINATION", "@finkouz"),
            review_destination=int(
                os.environ.get("TELEGRAM_REVIEW_CHAT_ID", "-5547276399")
            ),
        )
