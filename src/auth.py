"""Controle de usuários: registro, login e hash de senhas.

Senhas são armazenadas com PBKDF2-HMAC-SHA256 (240 mil iterações, salt
aleatório por usuário) — apenas a biblioteca padrão, sem dependências extras.
Formato armazenado: ``pbkdf2_sha256$<iterações>$<salt_hex>$<hash_hex>``.
"""

import hashlib
import hmac
import logging
import secrets

import database
from app_logging import log_event

PBKDF2_ITERATIONS = 240_000
MIN_USERNAME = 3
MIN_PASSWORD = 6


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iteracoes, salt, esperado = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iteracoes)
        )
        return hmac.compare_digest(digest.hex(), esperado)
    except (ValueError, TypeError):
        return False


def register(username: str, password: str) -> tuple[bool, str]:
    """Cria uma conta. Retorna (sucesso, mensagem para o usuário)."""
    username = username.strip()
    if len(username) < MIN_USERNAME:
        return False, f"O usuário precisa ter ao menos {MIN_USERNAME} caracteres."
    if len(password) < MIN_PASSWORD:
        return False, f"A senha precisa ter ao menos {MIN_PASSWORD} caracteres."

    user_id = database.create_user(username, hash_password(password))
    if user_id is None:
        return False, "Este nome de usuário já está em uso."

    log_event("user_register", f"Novo usuário registrado: '{username}'", user_id=user_id)
    return True, "Conta criada com sucesso! Faça login na aba 'Entrar'."


def login(username: str, password: str) -> tuple[dict | None, str]:
    """Autentica. Retorna (usuário, "") ou (None, mensagem de erro)."""
    username = username.strip()
    user = database.get_user_by_username(username)
    if user is None or not verify_password(password, user["password_hash"]):
        log_event("login_failed", f"Tentativa de login inválida para '{username}'",
                  level=logging.WARNING)
        return None, "Usuário ou senha inválidos."

    database.touch_last_login(user["id"])
    log_event("login", f"Login de '{username}'", user_id=user["id"])
    return user, ""
