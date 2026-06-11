"""Persistência local em SQLite: usuários, análises e logs de eventos.

Cada operação abre a própria conexão (modo WAL), o que é seguro com as
múltiplas threads de sessão do Streamlit sem precisar de pool ou locks.

O banco fica em `data/bee_tracker.db` (sobrescrevível via BEE_DB_PATH) —
em Docker, monte um volume em /app/data para persistir entre containers.
"""

import io
import os
import sqlite3
from contextlib import contextmanager

import numpy as np

DB_PATH = os.getenv("BEE_DB_PATH", os.path.join("data", "bee_tracker.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at        TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    video_name        TEXT NOT NULL,
    video_size_bytes  INTEGER,
    -- parâmetros usados na análise
    frame_skip        INTEGER,
    input_size        INTEGER,
    conf_threshold    REAL,
    nms_threshold     REAL,
    backend           TEXT,
    -- métricas do resultado
    fps_video         REAL,
    total_frames      INTEGER,
    frames_processed  INTEGER,
    detections        INTEGER,
    distance_px       REAL,
    avg_speed_px_s    REAL,
    processing_time_s REAL,
    fps_processing    REAL,
    pixels_per_mm     REAL,  -- escala de calibração (0/NULL = não calibrado)
    -- dados brutos para re-renderizar gráficos e exportar CSV
    centroids         BLOB,  -- np.save de array (N, 2)
    frame_indices     BLOB,  -- np.save de array (N,)
    first_frame_jpg   BLOB   -- primeiro frame comprimido em JPEG
);

CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id, created_at);

CREATE TABLE IF NOT EXISTS logs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    level   TEXT NOT NULL,
    event   TEXT NOT NULL,
    user_id INTEGER,
    message TEXT,
    context TEXT  -- JSON com detalhes do evento
);

CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts);
CREATE INDEX IF NOT EXISTS idx_logs_user ON logs(user_id);
"""


def _connect() -> sqlite3.Connection:
    diretorio = os.path.dirname(DB_PATH)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


def _migrate(conn):
    """Migrações idempotentes para bancos criados em versões anteriores."""
    colunas = {r["name"] for r in conn.execute("PRAGMA table_info(analyses)")}
    if "pixels_per_mm" not in colunas:
        conn.execute("ALTER TABLE analyses ADD COLUMN pixels_per_mm REAL")


# --- Serialização de arrays ---

def ndarray_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def blob_to_ndarray(blob) -> np.ndarray | None:
    if not blob:
        return None
    return np.load(io.BytesIO(blob))


# --- Usuários ---

def create_user(username: str, password_hash: str) -> int | None:
    """Cria o usuário; retorna o id ou None se o nome já existe."""
    try:
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_user_by_username(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def touch_last_login(user_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = datetime('now', 'localtime') WHERE id = ?",
            (user_id,),
        )


# --- Análises (sempre filtradas por user_id — isolamento entre usuários) ---

def save_analysis(user_id: int, video_name: str, video_size_bytes: int,
                  params: dict, metrics: dict,
                  centroids_blob: bytes, frame_indices_blob: bytes,
                  first_frame_jpg: bytes | None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO analyses (
                   user_id, video_name, video_size_bytes,
                   frame_skip, input_size, conf_threshold, nms_threshold, backend,
                   fps_video, total_frames, frames_processed, detections,
                   distance_px, avg_speed_px_s, processing_time_s, fps_processing,
                   pixels_per_mm,
                   centroids, frame_indices, first_frame_jpg
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                user_id, video_name, video_size_bytes,
                params.get("frame_skip"), params.get("input_size"),
                params.get("conf"), params.get("nms"), params.get("backend"),
                metrics.get("fps_video"), metrics.get("total_frames"),
                metrics.get("frames_processed"), metrics.get("detections"),
                metrics.get("distance_px"), metrics.get("avg_speed_px_s"),
                metrics.get("processing_time_s"), metrics.get("fps_processing"),
                params.get("pixels_per_mm"),
                centroids_blob, frame_indices_blob, first_frame_jpg,
            ),
        )
        return cur.lastrowid


def list_analyses(user_id: int) -> list[dict]:
    """Resumo das análises do usuário (sem os blobs), mais recentes primeiro."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, created_at, video_name, video_size_bytes,
                      frame_skip, input_size, conf_threshold, nms_threshold, backend,
                      fps_video, total_frames, frames_processed, detections,
                      distance_px, avg_speed_px_s, processing_time_s, fps_processing
               FROM analyses WHERE user_id = ? ORDER BY created_at DESC, id DESC""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_analysis(analysis_id: int, user_id: int) -> dict | None:
    """Análise completa (com blobs). Retorna None se não for do usuário."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def list_analyses_full(user_id: int) -> list[dict]:
    """Todas as análises do usuário com blobs (para gerar relatório consolidado)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at, id",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_analysis(analysis_id: int, user_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        )
        return cur.rowcount > 0


# --- Logs ---

def insert_log(level: str, event: str, message: str,
               user_id: int | None = None, context: str | None = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO logs (level, event, user_id, message, context) VALUES (?,?,?,?,?)",
            (level, event, user_id, message, context),
        )


def list_logs(user_id: int | None = None, limit: int = 100) -> list[dict]:
    with get_conn() as conn:
        if user_id is not None:
            rows = conn.execute(
                "SELECT ts, level, event, message FROM logs "
                "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts, level, event, user_id, message FROM logs "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
