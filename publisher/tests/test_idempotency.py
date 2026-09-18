from app.telegram_client import random_id_for


def test_random_id_is_stable_and_signed_64_bit():
    a = random_id_for("batch-123")
    b = random_id_for("batch-123")
    c = random_id_for("batch-124")
    assert a == b
    assert a != c
    assert -(2**63) <= a < 2**63
