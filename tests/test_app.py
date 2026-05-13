def test_app_imports():
    from app.main import app

    assert app.title == "Telegram AI Bot"


def test_telegram_webhook_route_registered():
    from app.main import app

    assert any(route.path == "/telegram/webhook" for route in app.routes)
