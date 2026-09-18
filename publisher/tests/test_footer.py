from app.footer import build_message, utf16_len


def test_utf16_len_counts_surrogate_pair_as_two_units():
    assert utf16_len("📱") == 2
    assert utf16_len("A📱B") == 4


def test_build_message_constructs_exact_footer_entities():
    text, entities = build_message("Test")
    assert text == "Test\n\n📱Instagram 📱App Store 📱Play Market 🌎 finko.uz"

    custom = [e for e in entities if e.kind == "custom_emoji"]
    links = [e for e in entities if e.kind == "text_url"]

    assert [e.document_id for e in custom] == [
        5319160079465857105,
        5370722600668382252,
        5373130604147654226,
        5785209342986817408,
    ]
    assert all(e.length == 2 for e in custom)
    assert [e.url for e in links] == [
        "https://www.instagram.com/finko_uz/",
        "https://apps.apple.com/uz/app/finko-finance-ko/id6755234580",
        "https://play.google.com/store/apps/details?id=com.finko&pcampaignid=web_share",
        "https://finko.uz/",
    ]
    first_emoji_offset = utf16_len("Test\n\n")
    assert custom[0].offset == first_emoji_offset
    assert links[0].offset == first_emoji_offset + 2
    assert links[0].length == utf16_len("Instagram")


def test_build_message_parses_supported_telegram_html_before_footer():
    body = '<b>Headline</b>\n\nText &amp; context. Manba: <a href="https://example.com/source">Markaziy bank</a> va <i>izoh</i>.'

    text, entities = build_message(body)

    assert text.startswith('Headline\n\nText & context. Manba: Markaziy bank va izoh.')
    assert '<b>' not in text
    assert '<a ' not in text
    assert '<i>' not in text

    bold = next(e for e in entities if e.kind == 'bold')
    assert bold.offset == 0
    assert bold.length == utf16_len('Headline')

    source_link = next(
        e for e in entities
        if e.kind == 'text_url' and e.url == 'https://example.com/source'
    )
    source_prefix = 'Headline\n\nText & context. Manba: '
    assert source_link.offset == utf16_len(source_prefix)
    assert source_link.length == utf16_len('Markaziy bank')

    italic = next(e for e in entities if e.kind == 'italic')
    italic_prefix = source_prefix + 'Markaziy bank va '
    assert italic.offset == utf16_len(italic_prefix)
    assert italic.length == utf16_len('izoh')
