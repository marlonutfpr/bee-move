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


def change_password(user_id: int, atual: str, nova: str) -> tuple[bool, str]:
    """Troca a senha do próprio usuário (exige a senha atual)."""
    user = database.get_user_by_id(user_id)
    if user is None:
        return False, "Usuário não encontrado."
    if not verify_password(atual, user["password_hash"]):
        return False, "A senha atual está incorreta."
    if len(nova) < MIN_PASSWORD:
        return False, f"A nova senha precisa ter ao menos {MIN_PASSWORD} caracteres."
    database.set_password(user_id, hash_password(nova))
    log_event("password_changed", "Senha alterada pelo próprio usuário", user_id=user_id)
    return True, "Senha alterada com sucesso."


# --- Operações de administrador ---

def admin_create_user(username: str, password: str, role: str = "user") -> tuple[bool, str]:
    username = username.strip()
    if len(username) < MIN_USERNAME:
        return False, f"O usuário precisa ter ao menos {MIN_USERNAME} caracteres."
    if len(password) < MIN_PASSWORD:
        return False, f"A senha precisa ter ao menos {MIN_PASSWORD} caracteres."
    if role not in ("user", "admin"):
        role = "user"
    user_id = database.create_user(username, hash_password(password), role=role)
    if user_id is None:
        return False, "Este nome de usuário já está em uso."
    log_event("user_created_admin", f"Usuário '{username}' criado pelo admin ({role})",
              user_id=user_id)
    return True, f"Usuário '{username}' criado."


def admin_reset_password(user_id: int, nova: str) -> tuple[bool, str]:
    if len(nova) < MIN_PASSWORD:
        return False, f"A nova senha precisa ter ao menos {MIN_PASSWORD} caracteres."
    database.set_password(user_id, hash_password(nova))
    log_event("password_reset_admin", f"Senha redefinida pelo admin (user {user_id})",
              user_id=user_id)
    return True, "Senha redefinida."


def admin_set_role(user_id: int, role: str) -> tuple[bool, str]:
    if role not in ("user", "admin"):
        return False, "Perfil inválido."
    # Impede remover o último admin do sistema.
    if role == "user":
        alvo = database.get_user_by_id(user_id)
        if alvo and alvo["role"] == "admin" and database.count_admins() <= 1:
            return False, "Não é possível rebaixar o último administrador."
    database.set_role(user_id, role)
    log_event("role_changed_admin", f"Perfil do usuário {user_id} alterado para {role}",
              user_id=user_id)
    return True, "Perfil atualizado."


def admin_delete_user(user_id: int) -> tuple[bool, str]:
    alvo = database.get_user_by_id(user_id)
    if alvo is None:
        return False, "Usuário não encontrado."
    if alvo["role"] == "admin" and database.count_admins() <= 1:
        return False, "Não é possível excluir o último administrador."
    database.delete_user(user_id)
    log_event("user_deleted_admin", f"Usuário '{alvo['username']}' excluído pelo admin",
              user_id=None)
    return True, f"Usuário '{alvo['username']}' excluído (e suas análises)."
