# FINKO Telegram AI Bot

FastAPI Telegram webhook bot with OpenAI Responses API, FINKO knowledge-base retrieval, analytics, three-language UI (Russian, Uzbek Latin, English), and a local loan calculator.

## Structure

```text
app/
├── main.py                 # FastAPI app, lifespan, router registration
├── config.py               # pydantic-settings config, env validation
├── db/
│   ├── connection.py       # aiosqlite connection, schema init
│   └── repositories.py     # users/messages/events repository functions
├── bot/
│   ├── handlers.py         # Telegram webhook flow
│   ├── language.py         # language detection and localized service messages
│   ├── ui.py               # keyboards and language selection
│   ├── intents.py          # intent and user-type detection
│   ├── quick_answers.py    # menu/quick replies
│   ├── calculator.py       # loan parsing and repayment schedules
│   ├── state.py            # Redis-backed state with in-memory fallback
│   └── telegram_update.py  # Telegram update parsing
├── kb/
│   └── search.py           # vector-store search and query expansion
├── services/
│   ├── openai_client.py    # shared AsyncOpenAI client
│   ├── openai_service.py   # prompting and answer generation
│   └── telegram.py         # shared httpx client and Telegram retry logic
├── analytics/
│   └── routes.py           # analytics API and dashboard
└── utils/
    └── retry.py            # shared retry helpers
```


## May 2026 update

- Uzbek Cyrillic was removed from the interface and knowledge-base upload list. Uzbek questions written in Cyrillic are still recognized, but answers are sent in Uzbek Latin.
- The FINKO partner list was updated: two outdated bank partners were removed; Aloqa Miqromoliya Tashkiloti, Una Moliya, and VAFO MOLIYA were added to MFO partners.
- E-IMZO knowledge was added in Russian, Uzbek Latin, and English, with a local quick answer for short questions such as `E-imzo kalit nima?`.
- After changing KB files, run `python kb_setup.py` and update `OPENAI_VECTOR_STORE_ID` in Railway/.env with the new vector store ID.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill .env values
uvicorn app.main:app --reload
```

## Tests

```bash
pytest -q
```

## Railway notes

Required env vars:

- `OPENAI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_SECRET`
- `OPENAI_VECTOR_STORE_ID`

Recommended for production:

- Add a Railway Redis plugin and set `REDIS_URL`.
- Without Redis the bot still works, but duplicate update cache, chat memory, calculator pending state, and rate limits are stored only in process memory.
- Keep one worker if Redis is not configured.

## Analytics

Analytics endpoints require `x-admin-token` header matching `ADMIN_ANALYTICS_TOKEN`.

- `/analytics/dashboard`
- `/analytics/summary`
- `/analytics/top-questions`
- `/analytics/top-users`
- `/analytics/recent-messages`
- `/analytics/export/messages.csv`
