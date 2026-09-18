from dataclasses import dataclass
from html.parser import HTMLParser


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


class _TelegramHTMLParser(HTMLParser):
    _KINDS = {
        "b": "bold",
        "i": "italic",
        "a": "text_url",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._position = 0
        self._entities: list[EntitySpec] = []
        self._stack: list[tuple[str, int, str | None]] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)
        self._position += utf16_len(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag not in self._KINDS:
            raise ValueError(f"unsupported_html_tag:{tag}")

        url = None
        if tag == "a":
            url = dict(attrs).get("href")
            if not url:
                raise ValueError("telegram_html_link_missing_href")

        self._stack.append((tag, self._position, url))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not self._stack or self._stack[-1][0] != tag:
            raise ValueError("malformed_telegram_html")

        open_tag, start, url = self._stack.pop()
        length = self._position - start
        if length <= 0:
            return

        self._entities.append(
            EntitySpec(
                kind=self._KINDS[open_tag],
                offset=start,
                length=length,
                url=url,
            )
        )

    def handle_startendtag(self, tag: str, attrs) -> None:
        raise ValueError(f"unsupported_html_tag:{tag.lower()}")

    def handle_comment(self, data: str) -> None:
        raise ValueError("unsupported_html_comment")

    def result(self) -> tuple[str, list[EntitySpec]]:
        if self._stack:
            raise ValueError("malformed_telegram_html")

        entities = sorted(
            self._entities,
            key=lambda entity: (entity.offset, -entity.length, entity.kind),
        )
        return "".join(self._parts), entities


def parse_telegram_html(value: str) -> tuple[str, list[EntitySpec]]:
    parser = _TelegramHTMLParser()
    parser.feed(value.rstrip())
    parser.close()
    return parser.result()


def build_message(body: str) -> tuple[str, list[EntitySpec]]:
    body_text, body_entities = parse_telegram_html(body)
    text = body_text + "\n\n"
    entities = list(body_entities)

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
