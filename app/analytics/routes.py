import csv
import io

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.config import ADMIN_ANALYTICS_TOKEN, logger
from app.db.connection import db_lock, get_db

router = APIRouter()

def verify_admin_token(x_admin_token: str | None) -> None:
    if not ADMIN_ANALYTICS_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ADMIN_ANALYTICS_TOKEN is not configured")
    if x_admin_token != ADMIN_ANALYTICS_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin token")


def verify_admin_token_flexible(header_token: str | None, query_token: str | None) -> None:
    token = header_token or query_token
    if not ADMIN_ANALYTICS_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ADMIN_ANALYTICS_TOKEN is not configured")
    if token != ADMIN_ANALYTICS_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin token")


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "Telegram AI Bot is running"}


@router.get("/health")
async def health():
    try:
        async with get_db() as conn:
            async with conn.execute("SELECT 1") as cursor:
                await cursor.fetchone()
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        return JSONResponse({"status": "degraded", "db": str(e)}, status_code=503)


@router.get("/analytics/summary")
async def analytics_summary(x_admin_token: str | None = Header(default=None)):
    verify_admin_token(x_admin_token)
    async with db_lock:
        async with get_db() as conn:
            async with conn.execute("SELECT COUNT(*) AS c FROM users") as cursor:
                users_count = (await cursor.fetchone())["c"]
            async with conn.execute("SELECT COUNT(*) AS c FROM messages") as cursor:
                messages_count = (await cursor.fetchone())["c"]
            async with conn.execute("SELECT COUNT(*) AS c FROM messages WHERE direction = 'inbound'") as cursor:
                inbound_count = (await cursor.fetchone())["c"]
            async with conn.execute("SELECT COUNT(*) AS c FROM messages WHERE direction = 'outbound'") as cursor:
                outbound_count = (await cursor.fetchone())["c"]
            async with conn.execute("SELECT intent, COUNT(*) AS c FROM messages WHERE direction = 'inbound' AND intent IS NOT NULL GROUP BY intent ORDER BY c DESC LIMIT 10") as cursor:
                top_intents_rows = await cursor.fetchall()
            async with conn.execute("SELECT language, COUNT(*) AS c FROM messages WHERE direction = 'inbound' AND language IS NOT NULL GROUP BY language ORDER BY c DESC") as cursor:
                top_languages_rows = await cursor.fetchall()
            async with conn.execute("SELECT user_type, COUNT(*) AS c FROM users GROUP BY user_type ORDER BY c DESC") as cursor:
                user_types_rows = await cursor.fetchall()
    return {
        "users_count": users_count,
        "messages_count": messages_count,
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        "top_intents": [{row["intent"]: row["c"]} for row in top_intents_rows],
        "languages": [{row["language"]: row["c"]} for row in top_languages_rows],
        "user_types": [{row["user_type"]: row["c"]} for row in user_types_rows],
    }


@router.get("/analytics/top-questions")
async def analytics_top_questions(limit: int = Query(default=10, ge=1, le=50), x_admin_token: str | None = Header(default=None)):
    verify_admin_token(x_admin_token)
    async with db_lock:
        async with get_db() as conn:
            async with conn.execute(
                "SELECT text, COUNT(*) AS c FROM messages WHERE direction = 'inbound' GROUP BY text ORDER BY c DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
    return {"top_questions": [{"question": row["text"], "count": row["c"]} for row in rows]}


@router.get("/analytics/top-users")
async def analytics_top_users(limit: int = Query(default=10, ge=1, le=50), x_admin_token: str | None = Header(default=None)):
    verify_admin_token(x_admin_token)
    async with db_lock:
        async with get_db() as conn:
            async with conn.execute(
                "SELECT chat_id, username, first_name, last_name, messages_count, user_type, last_language, selected_language FROM users ORDER BY messages_count DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
    return {"top_users": [dict(row) for row in rows]}


@router.get("/analytics/recent-messages")
async def analytics_recent_messages(limit: int = Query(default=50, ge=1, le=200), x_admin_token: str | None = Header(default=None)):
    verify_admin_token(x_admin_token)
    async with db_lock:
        async with get_db() as conn:
            async with conn.execute("""
                SELECT m.chat_id, u.username, u.first_name, u.last_name,
                       m.direction, m.text, m.language, m.intent, m.source, m.created_at
                FROM messages m LEFT JOIN users u ON u.chat_id = m.chat_id
                ORDER BY m.created_at DESC LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()
    return {"recent_messages": [dict(row) for row in rows]}


@router.get("/analytics/export/messages.csv")
async def analytics_export_messages_csv(x_admin_token: str | None = Header(default=None)):
    verify_admin_token(x_admin_token)
    async with db_lock:
        async with get_db() as conn:
            async with conn.execute("""
                SELECT m.chat_id, u.username, u.first_name, u.last_name,
                       m.direction, m.text, m.language, m.intent, m.source, m.created_at
                FROM messages m LEFT JOIN users u ON u.chat_id = m.chat_id
                ORDER BY m.created_at DESC
            """) as cursor:
                rows = await cursor.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["chat_id","username","first_name","last_name","direction","text","language","intent","source","created_at"])
    for row in rows:
        writer.writerow(list(row))

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=bot_messages_analytics.csv"},
    )


@router.get("/analytics/dashboard", response_class=HTMLResponse)
async def analytics_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>FINKO Bot Analytics</title>
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
            }
            .wrap {
                max-width: 1400px;
                margin: 0 auto;
                padding: 24px;
            }
            h1 {
                margin: 0 0 20px;
                font-size: 28px;
            }
            .topbar {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-bottom: 24px;
            }
            input {
                flex: 1;
                min-width: 280px;
                padding: 12px 14px;
                border-radius: 10px;
                border: 1px solid #334155;
                background: #111827;
                color: white;
                outline: none;
            }
            button, a.btn {
                padding: 12px 18px;
                border: none;
                border-radius: 10px;
                background: #2563eb;
                color: white;
                cursor: pointer;
                font-weight: 600;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }
            button:hover, a.btn:hover {
                background: #1d4ed8;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .card {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 16px;
                padding: 18px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            }
            .card h3 {
                margin: 0 0 8px;
                font-size: 14px;
                color: #94a3b8;
                font-weight: 600;
            }
            .value {
                font-size: 28px;
                font-weight: 700;
                color: #f8fafc;
            }
            .section {
                margin-top: 22px;
            }
            .section h2 {
                font-size: 20px;
                margin: 0 0 12px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background: #111827;
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid #1f2937;
            }
            th, td {
                padding: 12px 14px;
                border-bottom: 1px solid #1f2937;
                text-align: left;
                vertical-align: top;
                font-size: 14px;
            }
            th {
                background: #0b1220;
                color: #93c5fd;
            }
            tr:last-child td {
                border-bottom: none;
            }
            .muted {
                color: #94a3b8;
                font-size: 14px;
                margin-top: 8px;
            }
            .error {
                margin-top: 16px;
                padding: 12px 14px;
                border-radius: 10px;
                background: #7f1d1d;
                color: #fecaca;
                display: none;
            }
            .ok {
                margin-top: 16px;
                padding: 12px 14px;
                border-radius: 10px;
                background: #052e16;
                color: #bbf7d0;
                display: none;
            }
            .badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 999px;
                background: #1e293b;
                color: #cbd5e1;
                font-size: 12px;
            }
            .small {
                font-size: 13px;
                color: #94a3b8;
            }
            .message-cell {
                max-width: 520px;
                white-space: pre-wrap;
                word-break: break-word;
            }
        </style>
    </head>
    <body>
        <div class="wrap">
            <h1>FINKO Bot Analytics Dashboard</h1>

            <div class="topbar">
                <input
                    id="tokenInput"
                    type="password"
                    placeholder="Вставь ADMIN_ANALYTICS_TOKEN"
                />
                <button onclick="loadAnalytics()">Загрузить аналитику</button>
                <a href="#" class="btn" onclick="downloadCsv(event)">Скачать CSV</a>
            </div>

            <div class="muted">
                Доступно:
                <span class="badge">summary</span>
                <span class="badge">top questions</span>
                <span class="badge">top users</span>
                <span class="badge">recent messages</span>
                <span class="badge">csv export</span>
            </div>

            <div id="okBox" class="ok"></div>
            <div id="errorBox" class="error"></div>

            <div class="grid section">
                <div class="card">
                    <h3>Пользователи</h3>
                    <div class="value" id="usersCount">-</div>
                </div>
                <div class="card">
                    <h3>Всего сообщений</h3>
                    <div class="value" id="messagesCount">-</div>
                </div>
                <div class="card">
                    <h3>Входящие</h3>
                    <div class="value" id="inboundCount">-</div>
                </div>
                <div class="card">
                    <h3>Исходящие</h3>
                    <div class="value" id="outboundCount">-</div>
                </div>
            </div>

            <div class="section">
                <h2>Топ intent-ов</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Intent</th>
                            <th>Количество</th>
                        </tr>
                    </thead>
                    <tbody id="intentsTable"></tbody>
                </table>
            </div>

            <div class="section">
                <h2>Языки пользователей</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Язык</th>
                            <th>Количество</th>
                        </tr>
                    </thead>
                    <tbody id="languagesTable"></tbody>
                </table>
            </div>

            <div class="section">
                <h2>Типы пользователей</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Тип</th>
                            <th>Количество</th>
                        </tr>
                    </thead>
                    <tbody id="userTypesTable"></tbody>
                </table>
            </div>

            <div class="section">
                <h2>Топ вопросов</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Вопрос</th>
                            <th>Количество</th>
                        </tr>
                    </thead>
                    <tbody id="questionsTable"></tbody>
                </table>
            </div>

            <div class="section">
                <h2>Топ пользователей</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Chat ID</th>
                            <th>Username</th>
                            <th>Имя</th>
                            <th>Сообщений</th>
                            <th>Тип</th>
                            <th>Last language</th>
                            <th>UI language</th>
                        </tr>
                    </thead>
                    <tbody id="usersTable"></tbody>
                </table>
            </div>

            <div class="section">
                <h2>Кто что написал</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Дата и время</th>
                            <th>Chat ID</th>
                            <th>Username</th>
                            <th>Имя</th>
                            <th>Direction</th>
                            <th>Язык</th>
                            <th>Intent</th>
                            <th>Источник</th>
                            <th>Сообщение</th>
                        </tr>
                    </thead>
                    <tbody id="recentMessagesTable"></tbody>
                </table>
            </div>
        </div>

        <script>
            function showError(message) {
                const box = document.getElementById("errorBox");
                box.style.display = "block";
                box.textContent = message;
                document.getElementById("okBox").style.display = "none";
            }

            function showOk(message) {
                const box = document.getElementById("okBox");
                box.style.display = "block";
                box.textContent = message;
                document.getElementById("errorBox").style.display = "none";
            }

            function clearTables() {
                document.getElementById("intentsTable").innerHTML = "";
                document.getElementById("languagesTable").innerHTML = "";
                document.getElementById("userTypesTable").innerHTML = "";
                document.getElementById("questionsTable").innerHTML = "";
                document.getElementById("usersTable").innerHTML = "";
                document.getElementById("recentMessagesTable").innerHTML = "";
            }

            function escapeHtml(value) {
                return String(value ?? "").replace(/[&<>"']/g, char => ({
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;",
                    '"': "&quot;",
                    "'": "&#39;",
                }[char]));
            }

            function renderKeyValueRows(targetId, items) {
                const tbody = document.getElementById(targetId);
                tbody.innerHTML = "";

                if (!items || items.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="2" class="small">Нет данных</td></tr>';
                    return;
                }

                items.forEach(item => {
                    const key = Object.keys(item)[0];
                    const value = item[key];
                    const row = document.createElement("tr");
                    row.innerHTML = `<td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td>`;
                    tbody.appendChild(row);
                });
            }

            function renderQuestions(items) {
                const tbody = document.getElementById("questionsTable");
                tbody.innerHTML = "";

                if (!items || items.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="2" class="small">Нет данных</td></tr>';
                    return;
                }

                items.forEach(item => {
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td>${escapeHtml(item.question)}</td>
                        <td>${escapeHtml(item.count)}</td>
                    `;
                    tbody.appendChild(row);
                });
            }

            function renderUsers(items) {
                const tbody = document.getElementById("usersTable");
                tbody.innerHTML = "";

                if (!items || items.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" class="small">Нет данных</td></tr>';
                    return;
                }

                items.forEach(item => {
                    const fullName = ((item.first_name || "") + " " + (item.last_name || "")).trim();

                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td>${escapeHtml(item.chat_id)}</td>
                        <td>${escapeHtml(item.username)}</td>
                        <td>${escapeHtml(fullName)}</td>
                        <td>${escapeHtml(item.messages_count ?? 0)}</td>
                        <td>${escapeHtml(item.user_type)}</td>
                        <td>${escapeHtml(item.last_language)}</td>
                        <td>${escapeHtml(item.selected_language)}</td>
                    `;
                    tbody.appendChild(row);
                });
            }

            function renderRecentMessages(items) {
                const tbody = document.getElementById("recentMessagesTable");
                tbody.innerHTML = "";

                if (!items || items.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9" class="small">Нет данных</td></tr>';
                    return;
                }

                items.forEach(item => {
                    const fullName = ((item.first_name || "") + " " + (item.last_name || "")).trim();

                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td>${escapeHtml(item.created_at)}</td>
                        <td>${escapeHtml(item.chat_id)}</td>
                        <td>${escapeHtml(item.username)}</td>
                        <td>${escapeHtml(fullName)}</td>
                        <td>${escapeHtml(item.direction)}</td>
                        <td>${escapeHtml(item.language)}</td>
                        <td>${escapeHtml(item.intent)}</td>
                        <td>${escapeHtml(item.source)}</td>
                        <td class="message-cell">${escapeHtml(item.text)}</td>
                    `;
                    tbody.appendChild(row);
                });
            }

            async function fetchJson(url, token) {
                const response = await fetch(url, {
                    headers: {
                        "x-admin-token": token
                    }
                });

                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`HTTP ${response.status}: ${text}`);
                }

                return await response.json();
            }

            async function loadAnalytics() {
                const token = document.getElementById("tokenInput").value.trim();

                if (!token) {
                    showError("Сначала вставь ADMIN_ANALYTICS_TOKEN.");
                    return;
                }

                clearTables();

                try {
                    const [summary, questions, users, recentMessages] = await Promise.all([
                        fetchJson("/analytics/summary", token),
                        fetchJson("/analytics/top-questions?limit=15", token),
                        fetchJson("/analytics/top-users?limit=15", token),
                        fetchJson("/analytics/recent-messages?limit=100", token),
                    ]);

                    document.getElementById("usersCount").textContent = summary.users_count ?? 0;
                    document.getElementById("messagesCount").textContent = summary.messages_count ?? 0;
                    document.getElementById("inboundCount").textContent = summary.inbound_count ?? 0;
                    document.getElementById("outboundCount").textContent = summary.outbound_count ?? 0;

                    renderKeyValueRows("intentsTable", summary.top_intents || []);
                    renderKeyValueRows("languagesTable", summary.languages || []);
                    renderKeyValueRows("userTypesTable", summary.user_types || []);
                    renderQuestions(questions.top_questions || []);
                    renderUsers(users.top_users || []);
                    renderRecentMessages(recentMessages.recent_messages || []);

                    showOk("Аналитика успешно загружена.");
                } catch (error) {
                    showError("Не удалось загрузить аналитику: " + error.message);
                }
            }

            async function downloadCsv(event) {
                event.preventDefault();

                const token = document.getElementById("tokenInput").value.trim();

                if (!token) {
                    showError("Сначала вставь ADMIN_ANALYTICS_TOKEN.");
                    return;
                }

                try {
                    const response = await fetch("/analytics/export/messages.csv", {
                        headers: {
                            "x-admin-token": token
                        }
                    });

                    if (!response.ok) {
                        const text = await response.text();
                        throw new Error(`HTTP ${response.status}: ${text}`);
                    }

                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement("a");
                    link.href = url;
                    link.download = "bot_messages_analytics.csv";
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                    URL.revokeObjectURL(url);
                } catch (error) {
                    showError("Не удалось скачать CSV: " + error.message);
                }
            }
        </script>
    </body>
    </html>
    """
