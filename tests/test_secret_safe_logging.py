import io
import logging

from app.config import SecretRedactingFormatter, install_secret_safe_logging


def test_redacts_telegram_token_in_http_url_and_exception():
    token = "123456789:AA" + "x" * 32
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SecretRedactingFormatter(logging.Formatter("%(message)s %(exc_text)s")))
    logger = logging.getLogger("tests.safe-logging")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        try:
            raise RuntimeError(f"HTTP 401 at https://api.telegram.org/bot{token}/sendMessage")
        except RuntimeError:
            logger.exception("Request failed: https://api.telegram.org/bot%s/sendMessage", token)
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert token not in output
    assert "[REDACTED_TELEGRAM_TOKEN]" in output
    assert "sendMessage" in output


def test_httpx_routine_request_logging_is_disabled():
    install_secret_safe_logging()
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING


def test_logging_setup_is_idempotent():
    install_secret_safe_logging()
    install_secret_safe_logging()
    assert all(
        not isinstance(h.formatter, SecretRedactingFormatter)
        or not isinstance(h.formatter.wrapped, SecretRedactingFormatter)
        for h in logging.getLogger().handlers
    )
