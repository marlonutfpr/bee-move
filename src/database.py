"""Persistência local em SQLite: usuários, análises e logs de eventos.

Cada operação abre a própria conexão (modo WAL), o que é seguro com as
múltiplas threads de sessão do Streamlit sem precisar de pool ou locks.

O banco fica em `data/bee_tracker.db` (sobrescrevível via BEE_DB_PATH) —
em Docker, monte um volume em /app/data para persistir entre containers.
"""

import io
import json
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
    zones             TEXT,  -- JSON: áreas de monitoramento [{name, points}]
    num_targets       INTEGER,  -- nº de placas/abelhas rastreadas (1 = single)
    -- dados brutos para re-renderizar gráficos e exportar CSV
    centroids         BLOB,  -- np.save de array (M, 2)
    frame_indices     BLOB,  -- np.save de array (M,)
    track_ids         BLOB,  -- np.save de array (M,) — placa de cada centróide
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

-- Candidatos para reanotação (frames de detecções fracas / sem detecção),
-- usados para montar um dataset de fine-tuning.
CREATE TABLE IF NOT EXISTS annotation_candidates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    analysis_id INTEGER,
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    video_name  TEXT,
    area        TEXT,           -- nome da área recortada (quando houver zonas)
    frame_index INTEGER,
    reason      TEXT,           -- 'sem_deteccao' | 'baixa_confianca'
    pred_conf   REAL,           -- confiança da melhor detecção fraca (0 = nenhuma)
    pred_boxes  TEXT,           -- JSON: caixas sugeridas [[x,y,w,h,score], ...] (px)
    width       INTEGER,
    height      INTEGER,
    status      TEXT NOT NULL DEFAULT 'pendente',  -- pendente|anotado|sem_abelha|descartado
    labels      TEXT,           -- JSON: caixas anotadas [[x,y,w,h], ...] (px da imagem)
    image_jpg   BLOB
);
CREATE INDEX IF NOT EXISTS idx_cand_user ON annotation_candidates(user_id, status);

-- Configurações globais (ex.: modelo de detecção ativo).
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
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
    if "zones" not in colunas:
        conn.execute("ALTER TABLE analyses ADD COLUMN zones TEXT")
    if "num_targets" not in colunas:
        conn.execute("ALTER TABLE analyses ADD COLUMN num_targets INTEGER")
    if "track_ids" not in colunas:
        conn.execute("ALTER TABLE analyses ADD COLUMN track_ids BLOB")

    # Coluna 'area' nos candidatos (recorte por zona) — bancos anteriores.
    cols_cand = {r["name"] for r in conn.execute(
        "PRAGMA table_info(annotation_candidates)")}
    if cols_cand and "area" not in cols_cand:
        conn.execute("ALTER TABLE annotation_candidates ADD COLUMN area TEXT")

    # Perfis de usuário (admin/comum). Garante ao menos um admin nos bancos
    # antigos promovendo o usuário mais antigo.
    cols_users = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    if "role" not in cols_users:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
    sem_admin = conn.execute(
        "SELECT COUNT(*) AS c FROM users WHERE role = 'admin'"
    ).fetchone()["c"] == 0
    if sem_admin:
        primeiro = conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
        if primeiro:
            conn.execute("UPDATE users SET role = 'admin' WHERE id = ?", (primeiro["id"],))


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

def create_user(username: str, password_hash: str, role: str = None) -> int | None:
    """Cria o usuário; retorna o id ou None se o nome já existe.

    Sem `role` explícito, o PRIMEIRO usuário do sistema vira 'admin' e os
    demais 'user'.
    """
    try:
        with get_conn() as conn:
            if role is None:
                n = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
                role = "admin" if n == 0 else "user"
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role),
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


def get_user_by_id(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def touch_last_login(user_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = datetime('now', 'localtime') WHERE id = ?",
            (user_id,),
        )


# --- Administração de usuários ---

def list_users() -> list[dict]:
    """Todos os usuários com nº de análises, para o painel de administração."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT u.id, u.username, u.role, u.created_at, u.last_login_at,
                      (SELECT COUNT(*) FROM analyses a WHERE a.user_id = u.id)
                          AS n_analises
               FROM users u ORDER BY u.id"""
        ).fetchall()
        return [dict(r) for r in rows]


def count_admins() -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE role = 'admin'"
        ).fetchone()["c"]


def set_password(user_id: int, password_hash: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                     (password_hash, user_id))


def set_role(user_id: int, role: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))


def delete_user(user_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cur.rowcount > 0


# --- Análises (sempre filtradas por user_id — isolamento entre usuários) ---

def save_analysis(user_id: int, video_name: str, video_size_bytes: int,
                  params: dict, metrics: dict,
                  centroids_blob: bytes, frame_indices_blob: bytes,
                  first_frame_jpg: bytes | None,
                  track_ids_blob: bytes | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO analyses (
                   user_id, video_name, video_size_bytes,
                   frame_skip, input_size, conf_threshold, nms_threshold, backend,
                   fps_video, total_frames, frames_processed, detections,
                   distance_px, avg_speed_px_s, processing_time_s, fps_processing,
                   pixels_per_mm, zones, num_targets,
                   centroids, frame_indices, track_ids, first_frame_jpg
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                user_id, video_name, video_size_bytes,
                params.get("frame_skip"), params.get("input_size"),
                params.get("conf"), params.get("nms"), params.get("backend"),
                metrics.get("fps_video"), metrics.get("total_frames"),
                metrics.get("frames_processed"), metrics.get("detections"),
                metrics.get("distance_px"), metrics.get("avg_speed_px_s"),
                metrics.get("processing_time_s"), metrics.get("fps_processing"),
                params.get("pixels_per_mm"), params.get("zones"),
                metrics.get("num_targets"),
                centroids_blob, frame_indices_blob, track_ids_blob, first_frame_jpg,
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
                      distance_px, avg_speed_px_s, processing_time_s, fps_processing,
                      pixels_per_mm, num_targets
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


# --- Candidatos para reanotação / dataset de fine-tuning ---

def add_candidate(user_id, analysis_id, video_name, frame_index, reason,
                  pred_conf, pred_boxes, width, height, image_jpg,
                  area=None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO annotation_candidates (
                   user_id, analysis_id, video_name, area, frame_index, reason,
                   pred_conf, pred_boxes, width, height, image_jpg
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (user_id, analysis_id, video_name, area, frame_index, reason,
             pred_conf, json.dumps(pred_boxes or []), width, height, image_jpg),
        )
        return cur.lastrowid


def list_candidates(user_id: int, status: str | None = None) -> list[dict]:
    """Resumo dos candidatos (sem o JPEG), mais recentes primeiro."""
    q = ("SELECT id, created_at, video_name, area, frame_index, reason, pred_conf, "
         "status, width, height FROM annotation_candidates WHERE user_id = ?")
    args = [user_id]
    if status:
        q += " AND status = ?"
        args.append(status)
    q += " ORDER BY id DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def candidate_counts(user_id: int) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS c FROM annotation_candidates "
            "WHERE user_id = ? GROUP BY status", (user_id,),
        ).fetchall()
    return {r["status"]: r["c"] for r in rows}


def get_candidate(cand_id: int, user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM annotation_candidates WHERE id = ? AND user_id = ?",
            (cand_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def set_candidate_labels(cand_id: int, user_id: int, labels, status: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE annotation_candidates SET labels = ?, status = ? "
            "WHERE id = ? AND user_id = ?",
            (json.dumps(labels or []), status, cand_id, user_id),
        )
        return cur.rowcount > 0


def delete_candidate(cand_id: int, user_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM annotation_candidates WHERE id = ? AND user_id = ?",
            (cand_id, user_id),
        )
        return cur.rowcount > 0


def list_annotated_candidates(user_id: int) -> list[dict]:
    """Candidatos já anotados (com JPEG e labels) para exportar o dataset.

    Inclui 'anotado' (com caixas) e 'sem_abelha' (amostra negativa, sem caixas).
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM annotation_candidates WHERE user_id = ? "
            "AND status IN ('anotado', 'sem_abelha') ORDER BY id",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Configurações globais ---

def get_setting(key: str, default=None):
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
