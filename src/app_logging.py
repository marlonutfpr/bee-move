"""Estrutura de log da aplicação.

Cada evento é gravado em dois destinos:
- Arquivo rotativo `logs/bee_tracker.log` (5 MB x 3 arquivos) — diagnóstico.
- Tabela `logs` do SQLite — consultável pela aplicação (histórico por usuário).

Use `log_event(evento, mensagem, user_id=..., **contexto)` em vez de chamar
o logging diretamente; o contexto extra vira JSON na coluna `context`.

Eventos registrados pela aplicação:
    app_start, user_register, login, login_failed, logout, model_loaded,
    video_uploaded, analysis_started, analysis_completed, analysis_failed,
    analysis_deleted
"""

import json
import logging
import os
from logging.handlers import RotatingFileHandler

import database

LOG_DIR = os.getenv("BEE_LOG_DIR", "logs")
_LOGGER_NAME = "bee_tracker"


class _CamposPadrao(logging.Filter):
    """Garante os campos extras usados no formato do arquivo."""

    def filter(self, record):
        if not hasattr(record, "event"):
            record.event = "app"
        if not hasattr(record, "user_id"):
            record.user_id = "-"
        return True


class SQLiteLogHandler(logging.Handler):
    """Espelha cada registro na tabela `logs` do banco."""

    def emit(self, record):
        try:
            contexto = getattr(record, "context", None)
            user_id = getattr(record, "user_id", None)
            database.insert_log(
                level=record.levelname,
                event=getattr(record, "event", "app"),
                message=record.getMessage(),
                user_id=user_id if isinstance(user_id, int) else None,
                context=json.dumps(contexto, ensure_ascii=False, default=str)
                if contexto else None,
            )
        except Exception:
            self.handleError(record)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:  # já configurado (Streamlit reexecuta o script)
        return logger

    logger.setLevel(logging.INFO)
    logger.addFilter(_CamposPadrao())

    os.makedirs(LOG_DIR, exist_ok=True)
    arquivo = RotatingFileHandler(
        os.path.join(LOG_DIR, "bee_tracker.log"),
        maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    arquivo.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(event)s | user=%(user_id)s | %(message)s"
    ))
    logger.addHandler(arquivo)
    logger.addHandler(SQLiteLogHandler())
    return logger


def log_event(event: str, message: str, user_id: int | None = None,
              level: int = logging.INFO, **context):
    """Registra um evento estruturado no arquivo e no banco."""
    get_logger().log(level, message, extra={
        "event": event,
        "user_id": user_id if user_id is not None else "-",
        "context": context or None,
    })
