from __future__ import annotations

import json
import os
import uuid
from typing import Any

import psycopg


def _db_url() -> str | None:
    # Dedicated env var for chat storage; allows reusing the existing Postgres container/db.
    url = (os.getenv("CHAT_DATABASE_URL") or "").strip()
    return url or None


def _enabled() -> bool:
    return bool(_db_url())


def init_chat_schema() -> None:
    """
    Best-effort schema initialization.

    We avoid migrations for now: CREATE TABLE IF NOT EXISTS keeps this deployable
    in the current container setup.
    """
    if not _enabled():
        return

    url = _db_url()
    assert url is not None

    with psycopg.connect(url) as conn:
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

    url = _db_url()
    assert url is not None

    meta_json = json.dumps(metadata, ensure_ascii=False, default=str) if metadata else None
    with psycopg.connect(url) as conn:
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

