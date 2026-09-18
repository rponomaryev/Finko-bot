# FINKO Telegram Publisher - Design

Date: 2026-09-18

## Goal

After an editor approves a generated post in the existing Make.com review flow, publish that post to the public FINKO Telegram channel `@finkouz` through a Telegram user session hosted on Railway so the outgoing message can carry the same custom/Premium emoji entities and clickable links as the existing manual footer.

This change must not alter the existing collector execution model.

## Scope

In scope:

- Keep the current Railway `collector` service unchanged as a cron-based collection process.
- Add a separate long-running Railway service named `publisher`.
- Expose one authenticated HTTP endpoint from `publisher` for Make.com.
- Accept an already-approved Telegram post from Make.
- Build the footer server-side with the four known custom emoji IDs and four clickable text links.
- Publish only to the FINKO channel `@finkouz`.
- Return a machine-readable success/failure response to Make.
- Change the Make `Nashr qilish` route so it calls Railway instead of publishing directly through the Telegram Bot API.
- Keep the existing review/approval UX in Make.

Out of scope:

- Changing the collector schedule or collection logic.
- Redesigning the Make data store or adding retention/cleanup logic.
- Changing GPT/Claude prompts except where strictly necessary for the publish handoff.
- Publishing to arbitrary chats or channels.
- Any Figma integration.

## Existing State

The current Railway `collector` service:

- Start command: `python -m finko_collector collect-once`.
- Runs as a cron job and exits after each run.
- Has no HTTP service domain.
- Uses a persistent volume for collector state.
- Already has Telegram MTProto credentials/session in Railway environment variables.

The current Make flow:

- Generates and reviews a final Uzbek post.
- Uses the `FINKO Approval` scenario.
- Shows the draft in a review chat.
- Has a `Nashr qilish` action that currently publishes via a Telegram bot module.

Because the collector is ephemeral and cron-driven, it must not be converted into the HTTP publisher.

## Architecture

```
Railway collector
      |
      v
    Make
      |
      v
 GPT / Claude
      |
      v
 Review group
      |
  Nashr qilish
      |
      v
 Make HTTPS request
      |
      v
Railway publisher
      |
 Telegram MTProto
      |
      v
    @finkouz
```

The publisher is isolated from the collector so a publisher outage or deployment cannot stop scheduled collection.

## Publisher Service

The publisher will be a long-running Python HTTP service.

Responsibilities:

1. Expose a health endpoint for Railway.
2. Expose a single publish endpoint for Make.
3. Authenticate every publish request with a shared secret.
4. Validate the request schema and destination.
5. Connect with the existing Telegram user session.
6. Append the FINKO footer with exact entity offsets.
7. Send the final message to `@finkouz`.
8. Return the Telegram message ID and status.

The service must not expose generic Telegram-send functionality.

### Endpoint

`POST /publish`

Request body:

```json
{
  "batch_id": "string",
  "text": "approved post body"
}
```

Required header:

`X-FINKO-PUBLISH-KEY: <shared secret>`

The destination is not supplied by Make. It is fixed server-side to `@finkouz`.

Success response:

```json
{
  "ok": true,
  "batch_id": "...",
  "chat": "@finkouz",
  "message_id": 123
}
```

Failure response uses a non-2xx HTTP status and a stable error code without leaking credentials.

## Telegram Session

The publisher receives these values as Railway service variables:

- `TELEGRAM_API_ID`
- `TELEGRAM_API_HASH`
- `TELEGRAM_SESSION_STRING`

They should be configured as Railway reference variables to the existing collector variables where possible so secrets are not copied into code or Make.

The publisher must never log the session string, API hash, or shared publish key.

## Footer

The publisher appends one blank line and this logical footer:

- custom emoji `5319160079465857105` + linked text `Instagram`
- custom emoji `5370722600668382252` + linked text `App Store`
- custom emoji `5373130604147654226` + linked text `Play Market`
- custom emoji `5785209342986817408` + linked text `finko.uz`

Links:

- Instagram: `https://www.instagram.com/finko_uz/`
- App Store: `https://apps.apple.com/uz/app/finko-finance-ko/id6755234580`
- Play Market: `https://play.google.com/store/apps/details?id=com.finko&pcampaignid=web_share`
- Website: `https://finko.uz/`

The service constructs Telegram `MessageEntityCustomEmoji` and `MessageEntityTextUrl` entities directly rather than relying on HTML parsing.

The exact entity offsets are computed from the final Unicode string at runtime.

## Custom Emoji Compatibility

Before switching production publishing, the publisher must run a non-public validation using the connected user session.

Validation target: the account's Saved Messages.

The test must verify that all four custom emoji IDs can be sent from the current account and that Telegram returns the custom emoji entities in the sent message.

If Telegram rejects a custom emoji because the connected account is not permitted to send it, production publishing must not be switched over silently. The implementation should report which entity failed so the footer can be adjusted deliberately.

## Make.com Change

The existing approval flow remains unchanged until the editor presses `Nashr qilish`.

At that point:

1. Read the stored approved draft.
2. Call the Railway publisher `POST /publish` endpoint.
3. Pass only `batch_id` and the approved text body.
4. Treat only a successful 2xx response with `ok=true` as published.
5. Store the returned Telegram message ID using the existing workflow record fields.
6. Do not additionally run the old Telegram bot publish module.

The review preview may continue to use the Telegram bot.

## Authentication and Security

- A high-entropy shared secret is generated for Make -> Railway.
- The secret is stored in Railway and Make connection/config, never in message text.
- Constant-time comparison is used server-side.
- The destination channel is hard-coded.
- Request body size is limited to a Telegram-safe maximum.
- No arbitrary method, chat ID, username, or Telegram entity data is accepted from Make.
- Publisher logs include batch ID, result, latency, and Telegram message ID only.
- Sensitive values are redacted.

## Reliability

The endpoint is synchronous: Make receives success only after Telegram confirms the message send.

For duplicate Make retries, the publisher should use a deterministic MTProto `random_id` derived from `batch_id` where supported by the send path. If the library abstraction does not expose `random_id`, the implementation should call the raw MTProto send method.

This prevents a network retry from creating a second channel post for the same approved batch.

## Error Handling

Examples:

- 401: invalid Make authentication.
- 422: invalid or oversized request.
- 409: conflicting duplicate request if detectable.
- 502: Telegram rejected the send.
- 503: Telegram session unavailable.

Make must not mark the post as published unless Railway returns a confirmed Telegram message ID.

## Railway Deployment

Create a separate service named `publisher` in the existing `FINKO Content Collector` Railway project.

Characteristics:

- Long-running service, no cron schedule.
- Restart policy suitable for an HTTP service.
- Public Railway service domain for the Make webhook.
- Healthcheck endpoint.
- One replica initially.
- No shared collector volume required.
- Uses Telegram credentials/session via Railway variables.

The collector remains unchanged.

## Verification

Before production cutover:

1. Unit-test footer string and entity offsets.
2. Unit-test request authentication and validation.
3. Unit-test deterministic idempotency ID generation.
4. Deploy publisher to Railway.
5. Confirm healthcheck is green.
6. Send a custom-emoji validation message to Saved Messages.
7. Verify the returned Telegram message contains all four custom emoji entities and all four text-link entities.
8. Call the publisher endpoint with a non-production test payload where safe.
9. Update Make only after Railway validation passes.
10. Verify one real approved post from the review flow reaches `@finkouz` once, with correct formatting and links.

## Rollback

Rollback is simple:

- Re-enable the previous Make Telegram bot publish module.
- Disable the Railway HTTP publish call.
- Leave the `publisher` service stopped if necessary.

The collector is unaffected by both deployment and rollback.
