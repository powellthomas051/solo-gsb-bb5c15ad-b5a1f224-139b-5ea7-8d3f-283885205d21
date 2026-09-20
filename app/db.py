"""SQLite 持久化层：连接、建表与文档/修订/幂等请求/写入租约的读写。"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id               TEXT PRIMARY KEY,
    current_revision INTEGER NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS revisions (
    document_id   TEXT    NOT NULL REFERENCES documents(id),
    revision      INTEGER NOT NULL,
    content       TEXT    NOT NULL,
    patch         TEXT,
    base_revision INTEGER,
    request_id    TEXT,
    replayed      INTEGER NOT NULL DEFAULT 0,
    author        TEXT,
    created_at    TEXT    NOT NULL,
    PRIMARY KEY (document_id, revision)
);

CREATE TABLE IF NOT EXISTS requests (
    document_id TEXT    NOT NULL REFERENCES documents(id),
    request_id  TEXT    NOT NULL,
    fingerprint TEXT    NOT NULL,
    revision    INTEGER NOT NULL,
    changed     INTEGER NOT NULL,
    PRIMARY KEY (document_id, request_id)
);

CREATE TABLE IF NOT EXISTS contracts (
    document_id TEXT PRIMARY KEY REFERENCES documents(id),
    schema_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS releases (
    release_id  TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    results     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- 每个文档至多一行：当前/最近一次租约。释放不删行（released_at 置位），
-- fencing_token 由此行单调递增、永不回收；过期或释放后任何持有者都可接管。
CREATE TABLE IF NOT EXISTS leases (
    document_id   TEXT PRIMARY KEY REFERENCES documents(id),
    holder        TEXT    NOT NULL,
    fencing_token INTEGER NOT NULL,
    expires_at    TEXT    NOT NULL,
    acquired_at   TEXT    NOT NULL,
    renewed_at    TEXT    NOT NULL,
    released_at   TEXT
);
"""


def _fmt(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def utcnow() -> str:
    return _fmt(datetime.now(timezone.utc))


def utc_after(seconds: float) -> str:
    """当前服务端 UTC 时间 + *seconds* 秒；格式与 :func:`utcnow` 一致，可按字符串比较先后。"""
    return _fmt(datetime.now(timezone.utc) + timedelta(seconds=seconds))


def canonical(value: Any) -> str:
    """确定性 JSON 序列化：紧凑、键排序，用于幂等指纹与内容比较。"""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """立即获取写锁的事务，避免并发写入时的 deferred-lock 死锁。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


@contextmanager
def read_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """只读事务：WAL 模式下为一致快照读，不取写锁、不阻塞并发写入。

    跨文档引用解析在同一快照内固定根文档与所有 current 引用的修订，
    并发提交不会改变本次解析所见的内容。
    """
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def init_db() -> None:
    """启动时调用：确保目录存在并自动建表（幂等）。"""
    parent = os.path.dirname(os.path.abspath(settings.db_path))
    os.makedirs(parent, exist_ok=True)
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ---------- 读取 ----------

def get_document(conn: sqlite3.Connection, doc_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()


def get_revision(
    conn: sqlite3.Connection, doc_id: str, revision: int
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM revisions WHERE document_id = ? AND revision = ?",
        (doc_id, revision),
    ).fetchone()


def list_revisions(
    conn: sqlite3.Connection,
    doc_id: str,
    limit: int,
    offset: int,
    desc: bool = False,
) -> list[sqlite3.Row]:
    order = "DESC" if desc else "ASC"
    return conn.execute(
        f"""SELECT revision, base_revision, request_id, replayed, author, created_at
            FROM revisions
            WHERE document_id = ?
            ORDER BY revision {order}
            LIMIT ? OFFSET ?""",
        (doc_id, limit, offset),
    ).fetchall()


def get_request(
    conn: sqlite3.Connection, doc_id: str, request_id: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM requests WHERE document_id = ? AND request_id = ?",
        (doc_id, request_id),
    ).fetchone()


# ---------- 写入 ----------

def insert_document(
    conn: sqlite3.Connection, doc_id: str, content: Any, author: Optional[str], now: str
) -> int:
    """调用方需已开启写事务。插入文档与其修订 1，返回修订号。"""
    conn.execute(
        "INSERT INTO documents (id, current_revision, created_at, updated_at)"
        " VALUES (?, 1, ?, ?)",
        (doc_id, now, now),
    )
    conn.execute(
        """INSERT INTO revisions
               (document_id, revision, content, patch, base_revision,
                request_id, replayed, author, created_at)
           VALUES (?, 1, ?, NULL, NULL, NULL, 0, ?, ?)""",
        (doc_id, canonical(content), author, now),
    )
    return 1


def insert_revision(
    conn: sqlite3.Connection,
    doc_id: str,
    revision: int,
    content: Any,
    patch: Any,
    base_revision: int,
    request_id: str,
    replayed: bool,
    author: Optional[str],
    now: str,
) -> None:
    conn.execute(
        "UPDATE documents SET current_revision = ?, updated_at = ? WHERE id = ?",
        (revision, now, doc_id),
    )
    conn.execute(
        """INSERT INTO revisions
               (document_id, revision, content, patch, base_revision,
                request_id, replayed, author, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            revision,
            canonical(content),
            canonical(patch),
            base_revision,
            request_id,
            1 if replayed else 0,
            author,
            now,
        ),
    )


# ---------- 契约（JSON Schema 2020-12） ----------


def get_contract(conn: sqlite3.Connection, doc_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM contracts WHERE document_id = ?", (doc_id,)
    ).fetchone()


def upsert_contract(
    conn: sqlite3.Connection, doc_id: str, schema: Any, now: str
) -> None:
    """调用方需已开启写事务，且 *schema* 已通过自检并验证过当前文档。"""
    conn.execute(
        "INSERT INTO contracts (document_id, schema_json, updated_at)"
        " VALUES (?, ?, ?)"
        " ON CONFLICT(document_id) DO UPDATE SET"
        "   schema_json = excluded.schema_json, updated_at = excluded.updated_at",
        (doc_id, canonical(schema), now),
    )


def delete_contract(conn: sqlite3.Connection, doc_id: str) -> None:
    """调用方需已开启写事务。无契约时为 no-op。"""
    conn.execute("DELETE FROM contracts WHERE document_id = ?", (doc_id,))


def insert_request(
    conn: sqlite3.Connection,
    doc_id: str,
    request_id: str,
    fingerprint: str,
    revision: int,
    changed: bool,
) -> None:
    conn.execute(
        "INSERT INTO requests (document_id, request_id, fingerprint, revision, changed)"
        " VALUES (?, ?, ?, ?, ?)",
        (doc_id, request_id, fingerprint, revision, 1 if changed else 0),
    )


# ---------- 原子发布 ----------


def get_release(conn: sqlite3.Connection, release_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM releases WHERE release_id = ?", (release_id,)
    ).fetchone()


def insert_release(
    conn: sqlite3.Connection,
    release_id: str,
    fingerprint: str,
    results: Any,
    now: str,
) -> None:
    """调用方需已开启写事务。results 以规范化 JSON 保存，用于幂等重放。"""
    conn.execute(
        "INSERT INTO releases (release_id, fingerprint, results, created_at)"
        " VALUES (?, ?, ?, ?)",
        (release_id, fingerprint, canonical(results), now),
    )


# ---------- 临时写入租约（fencing token） ----------


def get_lease(conn: sqlite3.Connection, doc_id: str) -> Optional[sqlite3.Row]:
    """读取租约行（含已过期 / 已释放的历史行）；活性判断用 :func:`lease_is_active`。"""
    return conn.execute(
        "SELECT * FROM leases WHERE document_id = ?", (doc_id,)
    ).fetchone()


def lease_is_active(row: Optional[sqlite3.Row], now: str) -> bool:
    """租约行是否代表一个**未释放且未过期**的有效租约。

    *now* 为服务端 UTC（:func:`utcnow` 格式）；时间戳格式统一，
    先后关系按字符串比较即得。
    """
    return row is not None and row["released_at"] is None and row["expires_at"] > now


def upsert_lease(
    conn: sqlite3.Connection,
    doc_id: str,
    holder: str,
    fencing_token: int,
    expires_at: str,
    now: str,
) -> None:
    """调用方需已开启写事务。授予新租约（首次或接管过期/已释放租约）。

    重置 holder、fencing_token 与全部时间戳，并清除 released_at；
    fencing_token 由调用方按「旧 token + 1」计算后传入，本函数不复用旧值。
    """
    conn.execute(
        """INSERT INTO leases
               (document_id, holder, fencing_token, expires_at,
                acquired_at, renewed_at, released_at)
           VALUES (?, ?, ?, ?, ?, ?, NULL)
           ON CONFLICT(document_id) DO UPDATE SET
               holder        = excluded.holder,
               fencing_token = excluded.fencing_token,
               expires_at    = excluded.expires_at,
               acquired_at   = excluded.acquired_at,
               renewed_at    = excluded.renewed_at,
               released_at   = NULL""",
        (doc_id, holder, fencing_token, expires_at, now, now),
    )


def refresh_lease(
    conn: sqlite3.Connection, doc_id: str, expires_at: str, now: str
) -> None:
    """调用方需已开启写事务。续期 / 同持有者重复获取：仅刷新过期时间，token 不变。"""
    conn.execute(
        "UPDATE leases SET expires_at = ?, renewed_at = ? WHERE document_id = ?",
        (expires_at, now, doc_id),
    )


def release_lease(conn: sqlite3.Connection, doc_id: str, now: str) -> None:
    """调用方需已开启写事务。释放租约：保留行（fencing_token 计数不回收）。"""
    conn.execute(
        "UPDATE leases SET released_at = ? WHERE document_id = ?",
        (now, doc_id),
    )
