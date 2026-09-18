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
