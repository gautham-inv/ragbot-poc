from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg


def _db_url() -> str | None:
    # Dedicated env var for chat storage; allows reusing the existing Postgres container/db.
    url = (os.getenv("CHAT_DATABASE_URL") or "").strip()
    return url or None


def _enabled() -> bool:
    return bool(_db_url())


# --------------------------------------------------------------------------
# Connection handling.
#
# Every call previously opened a fresh psycopg.connect() (two per chat turn) —
# a full TCP+auth handshake each time. Set CHAT_DB_POOL=1 to reuse a connection
# pool instead. Default (unset) keeps the exact previous behaviour: a direct
# connection per call. The pool path falls back to a direct connection if
# psycopg_pool isn't installed or the pool can't be created.
# --------------------------------------------------------------------------
_pool = None
_pool_disabled = False


def _pooling_enabled() -> bool:
    return (os.getenv("CHAT_DB_POOL") or "").strip() == "1"


def _get_pool():
    global _pool, _pool_disabled
    if _pool is not None or _pool_disabled:
        return _pool
    url = _db_url()
    if not url:
        return None
    try:
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(
            conninfo=url,
            min_size=int(os.getenv("CHAT_DB_POOL_MIN", "1")),
            max_size=int(os.getenv("CHAT_DB_POOL_MAX", "5")),
            open=True,
        )
        print("[chat_db] connection pool enabled.")
        return _pool
    except Exception as e:
        print(f"[chat_db] pool unavailable ({e}); using direct connections.")
        _pool_disabled = True
        return None


@contextmanager
def _connect(*, autocommit: bool = False) -> Iterator[psycopg.Connection[Any]]:
    """Yield a connection, from the pool when enabled, else a direct one.

    Commit-on-exit semantics match `psycopg.connect()` used as a context manager,
    whether pooled or direct.
    """
    url = _db_url()
    assert url is not None
    pool = _get_pool() if _pooling_enabled() else None
    if pool is not None:
        with pool.connection() as conn:
            if conn.autocommit != autocommit:
                conn.autocommit = autocommit
            yield conn
    else:
        with psycopg.connect(url, autocommit=autocommit) as conn:
            yield conn


_chat_schema_ready = False


def init_chat_schema() -> None:
    """
    Best-effort schema initialization.

    We avoid migrations for now: CREATE TABLE IF NOT EXISTS keeps this deployable
    in the current container setup.
    """
    global _chat_schema_ready

    if not _enabled():
        print("[chat_db] skipping init: CHAT_DATABASE_URL not set.")
        return

    url = _db_url()
    assert url is not None
    # Log the destination host for visibility in logs
    host = url.split("@")[-1] if "@" in url else "unknown"
    print(f"[chat_db] initializing schema at {host}...")

    try:
        # We use autocommit=True to ensure each DDL statement is committed immediately.
        with _connect(autocommit=True) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_conversations (
                  id uuid PRIMARY KEY,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  updated_at timestamptz NOT NULL DEFAULT now()
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                  id bigserial PRIMARY KEY,
                  conversation_id uuid NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
                  role text NOT NULL,
                  content text NOT NULL,
                  metadata jsonb NULL,
                  created_at timestamptz NOT NULL DEFAULT now()
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chat_messages_conversation_id_created_at_idx ON chat_messages (conversation_id, created_at);"
            )

            # Immediate verification
            res = conn.execute(
                "SELECT table_schema, table_name FROM information_schema.tables WHERE table_name = 'chat_conversations'"
            ).fetchone()
            if res:
                print(f"[chat_db] verification: table '{res[1]}' confirmed in schema '{res[0]}'.")
            else:
                print("[chat_db] verification FAILED: table 'chat_conversations' not found immediately after creation.")

        print("[chat_db] schema initialized successfully.")
        _chat_schema_ready = True
    except Exception as e:
        print(f"[chat_db] schema init failed: {e}")
        # We re-raise so the startup event can log it as a warning/error
        raise e


def _ensure_chat_schema_once() -> None:
    """
    Ensure chat schema exists. Safe to call before DB operations.
    """
    if not _enabled():
        return
    global _chat_schema_ready
    if _chat_schema_ready:
        return
    init_chat_schema()


def ensure_conversation_id(conversation_id: str | None) -> str:
    if conversation_id:
        try:
            uuid.UUID(conversation_id)
            return conversation_id
        except Exception:
            pass
    return str(uuid.uuid4())


def _ensure_conversation_row(conn: psycopg.Connection[Any], conversation_id: str) -> None:
    conn.execute(
        """
        INSERT INTO chat_conversations (id)
        VALUES (%s)
        ON CONFLICT (id) DO NOTHING;
        """,
        (conversation_id,),
    )


def insert_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Insert a message into Postgres. No-op when CHAT_DATABASE_URL isn't set.
    """
    if not _enabled():
        return
    _ensure_chat_schema_once()

    url = _db_url()
    assert url is not None

    meta_json = json.dumps(metadata, ensure_ascii=False, default=str) if metadata else None
    try:
        with _connect() as conn:
            _ensure_conversation_row(conn, conversation_id)
            conn.execute(
                """
                INSERT INTO chat_messages (conversation_id, role, content, metadata)
                VALUES (%s, %s, %s, %s::jsonb);
                """,
                (conversation_id, role, content or "", meta_json),
            )
            conn.execute(
                "UPDATE chat_conversations SET updated_at = now() WHERE id = %s;",
                (conversation_id,),
            )
    except psycopg.errors.UndefinedTable:
        # Self-heal once if tables are unexpectedly missing (e.g. external schema resets).
        init_chat_schema()
        with _connect() as conn:
            _ensure_conversation_row(conn, conversation_id)
            conn.execute(
                """
                INSERT INTO chat_messages (conversation_id, role, content, metadata)
                VALUES (%s, %s, %s, %s::jsonb);
                """,
                (conversation_id, role, content or "", meta_json),
            )
            conn.execute(
                "UPDATE chat_conversations SET updated_at = now() WHERE id = %s;",
                (conversation_id,),
            )


def list_conversations(*, limit: int = 50) -> list[dict[str, Any]]:
    """
    Return recent conversations with lightweight preview fields.
    """
    if not _enabled():
        return []
    _ensure_chat_schema_once()

    url = _db_url()
    assert url is not None

    safe_limit = max(1, min(int(limit), 200))
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  c.id::text AS id,
                  c.created_at,
                  c.updated_at,
                  (
                    SELECT m.content
                    FROM chat_messages m
                    WHERE m.conversation_id = c.id AND m.role = 'user'
                    ORDER BY m.created_at ASC, m.id ASC
                    LIMIT 1
                  ) AS first_user_message,
                  (
                    SELECT count(*)
                    FROM chat_messages m2
                    WHERE m2.conversation_id = c.id
                  )::int AS message_count
                FROM chat_conversations c
                ORDER BY c.updated_at DESC
                LIMIT %s;
                """,
                (safe_limit,),
            ).fetchall()
    except psycopg.errors.UndefinedTable:
        init_chat_schema()
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  c.id::text AS id,
                  c.created_at,
                  c.updated_at,
                  (
                    SELECT m.content
                    FROM chat_messages m
                    WHERE m.conversation_id = c.id AND m.role = 'user'
                    ORDER BY m.created_at ASC, m.id ASC
                    LIMIT 1
                  ) AS first_user_message,
                  (
                    SELECT count(*)
                    FROM chat_messages m2
                    WHERE m2.conversation_id = c.id
                  )::int AS message_count
                FROM chat_conversations c
                ORDER BY c.updated_at DESC
                LIMIT %s;
                """,
                (safe_limit,),
            ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": str(r[0]),
                "created_at": r[1].isoformat() if r[1] is not None else None,
                "updated_at": r[2].isoformat() if r[2] is not None else None,
                "first_user_message": str(r[3]) if r[3] is not None else None,
                "message_count": int(r[4] or 0),
            }
        )
    return out


def get_conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
    """
    Return all messages for a conversation in chronological order.
    """
    if not _enabled():
        return []
    _ensure_chat_schema_once()

    url = _db_url()
    assert url is not None

    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, metadata, created_at
                FROM chat_messages
                WHERE conversation_id = %s
                ORDER BY created_at ASC, id ASC;
                """,
                (conversation_id,),
            ).fetchall()
    except psycopg.errors.UndefinedTable:
        init_chat_schema()
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, metadata, created_at
                FROM chat_messages
                WHERE conversation_id = %s
                ORDER BY created_at ASC, id ASC;
                """,
                (conversation_id,),
            ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": int(r[0]),
                "role": str(r[1]),
                "content": str(r[2] or ""),
                "metadata": r[3] if isinstance(r[3], dict) else None,
                "created_at": r[4].isoformat() if r[4] is not None else None,
            }
        )
    return out

