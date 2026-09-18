from dataclasses import dataclass


@dataclass(frozen=True)
class EntitySpec:
    kind: str
    offset: int
    length: int
    document_id: int | None = None
    url: str | None = None


@dataclass(frozen=True)
class FooterItem:
    placeholder: str
    document_id: int
    label: str
    url: str


ITEMS = (
    FooterItem("📱", 5319160079465857105, "Instagram", "https://www.instagram.com/finko_uz/"),
    FooterItem("📱", 5370722600668382252, "App Store", "https://apps.apple.com/uz/app/finko-finance-ko/id6755234580"),
    FooterItem("📱", 5373130604147654226, "Play Market", "https://play.google.com/store/apps/details?id=com.finko&pcampaignid=web_share"),
    FooterItem("🌎", 5785209342986817408, "finko.uz", "https://finko.uz/"),
)


def utf16_len(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def build_message(body: str) -> tuple[str, list[EntitySpec]]:
    text = body.rstrip() + "\n\n"
    entities: list[EntitySpec] = []

    for index, item in enumerate(ITEMS):
        if index:
            text += " "

        emoji_offset = utf16_len(text)
        text += item.placeholder
        entities.append(
            EntitySpec(
                kind="custom_emoji",
                offset=emoji_offset,
                length=utf16_len(item.placeholder),
                document_id=item.document_id,
            )
        )

        if item.label == "finko.uz":
            text += " "

        label_offset = utf16_len(text)
        text += item.label
        entities.append(
            EntitySpec(
                kind="text_url",
                offset=label_offset,
                length=utf16_len(item.label),
                url=item.url,
            )
        )

    return text, entities
