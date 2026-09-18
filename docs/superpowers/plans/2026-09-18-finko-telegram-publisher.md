# FINKO Telegram Publisher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy an isolated Railway HTTP publisher that receives an approved FINKO post from Make, appends the exact four Telegram custom emoji entities plus clickable links, and publishes once to `@finkouz` through the existing Telegram user session.

**Architecture:** Add a standalone `publisher/` Python app inside `rponomaryev/Finko-bot`; Railway builds only that subdirectory as a new service. The current cron `collector` remains unchanged. Make's existing `Nashr qilish` route switches from direct Bot API publishing to an authenticated HTTPS call to the publisher only after the publisher passes Saved Messages custom-emoji validation.

**Tech Stack:** Python 3.13, FastAPI, Uvicorn, Telethon MTProto, pytest, Railway, Make.com.

**Spec:** `docs/superpowers/specs/2026-09-18-finko-telegram-publisher-design.md`

## Global Constraints

- Do not modify the existing Railway `collector` service's cron schedule, start command, volume, or collection logic.
- Do not redesign or clean the Make Data Store as part of this change.
- Public destination is fixed server-side to `@finkouz`; Make must not be able to choose an arbitrary chat.
- Custom emoji IDs: Instagram `5319160079465857105`; App Store `5370722600668382252`; Play Market `5373130604147654226`; Website `5785209342986817408`.
- Footer links are fixed server-side to the supplied FINKO URLs.
- Telegram entity offsets must be UTF-16 code-unit offsets, not Python code-point indexes.
- Production Make cutover must not happen until a Saved Messages probe proves that the current Telegram user session can send all four custom emoji entities.
- Do not log Telegram session strings, API hash, publish secret, or full post text.

---

## File Structure

Create:
- `publisher/app/__init__.py`
- `publisher/app/config.py`
- `publisher/app/footer.py`
- `publisher/app/telegram_client.py`
- `publisher/app/main.py`
- `publisher/requirements.txt`
- `publisher/tests/test_footer.py`
- `publisher/tests/test_api.py`
- `publisher/tests/test_idempotency.py`

No root application files need to change.

---

### Task 1: Footer and UTF-16 Entity Builder

**Files:**
- Create: `publisher/app/__init__.py`
- Create: `publisher/app/footer.py`
- Create: `publisher/tests/test_footer.py`

**Interfaces:**
- Produces: `build_message(body: str) -> tuple[str, list]`
- Produces: `utf16_len(value: str) -> int`

- [ ] **Step 1: Write the failing test**

```python
from telethon.tl.types import MessageEntityCustomEmoji, MessageEntityTextUrl
from app.footer import build_message, utf16_len

def test_utf16_len_counts_surrogate_pair_as_two_units():
    assert utf16_len("📱") == 2
    assert utf16_len("A📱B") == 4

def test_build_message_constructs_exact_footer_entities():
    text, entities = build_message("Test")
    assert text == "Test\n\n📱Instagram 📱App Store 📱Play Market 🌎 finko.uz"

    custom = [e for e in entities if isinstance(e, MessageEntityCustomEmoji)]
    links = [e for e in entities if isinstance(e, MessageEntityTextUrl)]

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
```

- [ ] **Step 2: Run RED**

Run: `cd publisher && pytest tests/test_footer.py -q`

Expected: FAIL because `app.footer` does not exist.

- [ ] **Step 3: Implement minimal builder**

```python
from dataclasses import dataclass
from telethon.tl.types import MessageEntityCustomEmoji, MessageEntityTextUrl

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

def build_message(body: str):
    text = body.rstrip() + "\n\n"
    entities = []
    for index, item in enumerate(ITEMS):
        if index:
            text += " "
        emoji_offset = utf16_len(text)
        text += item.placeholder
        entities.append(MessageEntityCustomEmoji(
            offset=emoji_offset,
            length=utf16_len(item.placeholder),
            document_id=item.document_id,
        ))
        if item.label == "finko.uz":
            text += " "
        label_offset = utf16_len(text)
        text += item.label
        entities.append(MessageEntityTextUrl(
            offset=label_offset,
            length=utf16_len(item.label),
            url=item.url,
        ))
    return text, entities
```

- [ ] **Step 4: Run GREEN**

Run: `cd publisher && pytest tests/test_footer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

`git commit -am "feat: build Telegram custom emoji footer"`

---

### Task 2: Telegram Session Client, Idempotency, and Saved Messages Probe

**Files:**
- Create: `publisher/app/config.py`
- Create: `publisher/app/telegram_client.py`
- Create: `publisher/tests/test_idempotency.py`

**Interfaces:**
- Consumes: `build_message(body)`
- Produces: `random_id_for(batch_id: str) -> int`
- Produces: `TelegramPublisher.publish(batch_id: str, body: str) -> int`
- Produces: `TelegramPublisher.probe_custom_emojis() -> ProbeResult`

- [ ] **Step 1: Write the failing deterministic-ID test**

```python
from app.telegram_client import random_id_for

def test_random_id_is_stable_and_signed_64_bit():
    a = random_id_for("batch-123")
    b = random_id_for("batch-123")
    c = random_id_for("batch-124")
    assert a == b
    assert a != c
    assert -(2**63) <= a < 2**63
```

- [ ] **Step 2: Run RED**

Run: `cd publisher && pytest tests/test_idempotency.py -q`

Expected: FAIL because `app.telegram_client` does not exist.

- [ ] **Step 3: Implement settings**

```python
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    session_string: str
    publish_key: str
    destination: str = "@finkouz"

    @classmethod
    def from_env(cls):
        return cls(
            api_id=int(os.environ["TELEGRAM_API_ID"]),
            api_hash=os.environ["TELEGRAM_API_HASH"],
            session_string=os.environ["TELEGRAM_SESSION_STRING"],
            publish_key=os.environ["FINKO_PUBLISH_KEY"],
        )
```

- [ ] **Step 4: Implement the Telethon raw MTProto client**

Use `StringSession`, `functions.messages.SendMessageRequest`, `MessageEntityCustomEmoji`, and the entities from Task 1. Derive the signed 64-bit `random_id` with BLAKE2b over `batch_id`. Provide a Saved Messages probe that sends and reads back the four entity IDs.

If `TELEGRAM_SESSION_STRING` is not accepted by Telethon, stop and adapt to the actual existing session format; do not regenerate or invalidate the collector session.

- [ ] **Step 5: Run GREEN**

Run: `cd publisher && pytest tests/test_idempotency.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

`git commit -am "feat: add MTProto Telegram publisher"`

---

### Task 3: Authenticated FastAPI Surface

**Files:**
- Create: `publisher/app/main.py`
- Create: `publisher/tests/test_api.py`
- Create: `publisher/requirements.txt`

**Interfaces:**
- Produces: `GET /health`
- Produces: `POST /publish`

- [ ] **Step 1: Write failing API tests**

```python
from fastapi.testclient import TestClient
import app.main as main

class FakePublisher:
    async def publish(self, batch_id: str, body: str) -> int:
        return 777

def test_publish_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(main, "publisher", FakePublisher())
    monkeypatch.setattr(main, "PUBLISH_KEY", "secret")
    client = TestClient(main.app)
    response = client.post("/publish", json={"batch_id": "b1", "text": "Hello"})
    assert response.status_code == 401

def test_publish_returns_confirmed_message_id(monkeypatch):
    monkeypatch.setattr(main, "publisher", FakePublisher())
    monkeypatch.setattr(main, "PUBLISH_KEY", "secret")
    client = TestClient(main.app)
    response = client.post(
        "/publish",
        headers={"X-FINKO-PUBLISH-KEY": "secret"},
        json={"batch_id": "b1", "text": "Hello"},
    )
    assert response.status_code == 200
    assert response.json()["message_id"] == 777
```

- [ ] **Step 2: Run RED**

Run: `cd publisher && pytest tests/test_api.py -q`

Expected: FAIL because `app.main` does not exist.

- [ ] **Step 3: Implement API**

Use `hmac.compare_digest` for `X-FINKO-PUBLISH-KEY`; `batch_id` length 1-160; `text` length 1-3600; hard-code output chat to `@finkouz`; return Telegram `message_id`; return 401 for auth failure, 422 for invalid input, 502 for Telegram send failure; never log secret or body.

- [ ] **Step 4: Pin dependencies**

```text
fastapi==0.128.2
uvicorn[standard]==0.46.0
telethon==1.42.0
pydantic==2.12.5
pytest==8.3.5
httpx==0.28.1
```

- [ ] **Step 5: Run all publisher tests**

Run: `cd publisher && pytest -q`

Expected: all PASS.

- [ ] **Step 6: Commit**

`git commit -am "feat: expose authenticated publish API"`

---

### Task 4: Deploy Isolated Railway Publisher and Validate Custom Emoji Support

**Files:** Railway configuration only.

- [ ] **Step 1: Create Railway service**

Project: `FINKO Content Collector`
Environment: `production`
GitHub source: `rponomaryev/Finko-bot`
Branch: `main`
Service name: `publisher`

- [ ] **Step 2: Configure deployment**

```text
rootDirectory=/publisher
startCommand=uvicorn app.main:app --host 0.0.0.0 --port $PORT
healthcheckPath=/health
restartPolicyType=ON_FAILURE
cronSchedule=null
```

- [ ] **Step 3: Configure variables using Railway references**

```text
TELEGRAM_API_ID=${{collector.TELEGRAM_API_ID}}
TELEGRAM_API_HASH=${{collector.TELEGRAM_API_HASH}}
TELEGRAM_SESSION_STRING=${{collector.TELEGRAM_SESSION_STRING}}
```

Generate and set a fresh high-entropy `FINKO_PUBLISH_KEY`.

- [ ] **Step 4: Verify deployment**

Poll until publisher deployment status is `SUCCESS`; inspect logs if `FAILED` or `CRASHED`.

- [ ] **Step 5: Verify health**

`GET https://<publisher-domain>/health` must return `{"ok": true}`.

- [ ] **Step 6: Run Saved Messages emoji probe**

Pass only if Telegram accepts the message and read-back contains all four exact custom emoji IDs. If Telegram rejects them because the connected user account lacks permission to send those custom emojis, stop before Make cutover. In that case, buying Telegram Premium on the same Telegram account behind `TELEGRAM_SESSION_STRING` is the simplest remediation; rerun the probe after activation.

- [ ] **Step 7: Verify collector unchanged**

Collector must still use `python -m finko_collector collect-once` and cron `0,30 4,8,12,16 * * *`.

---

### Task 5: Switch Make `Nashr qilish` to Railway Publisher

**Files:** Make scenario `FINKO Approval`.

- [ ] **Step 1: Add authenticated HTTP request after stored draft retrieval**

```text
POST https://<publisher-domain>/publish
Content-Type: application/json
X-FINKO-PUBLISH-KEY: <secret>
```

Body:

```json
{
  "batch_id": "{{get(split(2.callback_query.data; \"|\"); 2)}}",
  "text": "{{30.payload_json}}"
}
```

- [ ] **Step 2: Require confirmed publish result**

Continue only for HTTP 200, `ok=true`, and integer `message_id`.

- [ ] **Step 3: Store destination message ID**

Map Railway `message_id` into the existing workflow `destination_message_id`.

- [ ] **Step 4: Remove direct Telegram Bot API channel publish from this branch**

Keep review preview and approval buttons unchanged.

- [ ] **Step 5: Read back Make configuration**

Verify publish branch calls Railway once and there is no second direct channel send.

---

### Task 6: End-to-End Verification

- [ ] **Step 1:** Create a normal draft through the existing GPT/Claude flow.
- [ ] **Step 2:** Verify review preview and approval buttons.
- [ ] **Step 3:** Approve one safe real post with `Nashr qilish`.
- [ ] **Step 4:** Verify Make received `ok=true` and Telegram `message_id`.
- [ ] **Step 5:** Inspect `@finkouz`: exactly one post, four real custom/Premium emoji, four clickable links, no link previews, unchanged body.
- [ ] **Step 6:** Replay the same `batch_id` in a controlled test; deterministic MTProto `random_id` must prevent a duplicate channel post.
- [ ] **Step 7:** Verify collector latest scheduled run still succeeds, Make scenario remains active, publisher health is green, and logs contain no secrets.
