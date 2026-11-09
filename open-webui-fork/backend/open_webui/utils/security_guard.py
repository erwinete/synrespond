"""Единый контур безопасности synrespond: защита от SSRF, Path Traversal, ReDoS и утечки секретов.

SEC-01: безопасный разбор stdio-команд харнесов (shell=False + белый список бинарников).
SEC-02: валидация исходящих URL (вебхуки, HTTP-харнесы) против loopback/link-local/RFC1918.
SEC-03: безопасные подкаталоги хранилищ (model_id, collection_id).
SEC-06: ReDoS-фильтр пользовательских regex слотов.
8.1:   обратимое Fernet-шифрование auth_token харнесов на ключе WEBUI_SECRET_KEY.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import re
import shlex
import socket
from urllib.parse import urlparse

try:  # cryptography входит в зависимости ядра, но не делаем импорт критичным для тестов
    from cryptography.fernet import Fernet, InvalidToken
except Exception:  # pragma: no cover - fallback для окружений без cryptography
    Fernet = None
    InvalidToken = Exception

_DEFAULT_ALLOWED_HOSTS = {
    "encoder-classifier",
    "ragas-worker",
    "voice-sip-bridge",
    "speaches",
    "tei-embeddings",
    "tei-reranker",
    "vllm-server",
}

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_REDOS_NESTED_QUANTIFIER_RE = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*?{]")

MAX_SLOT_INPUT_LEN = 512


def get_allowed_internal_hosts() -> set[str]:
    extra = os.getenv("OUTBOUND_ALLOWED_HOSTS", "")
    custom = {h.strip().lower() for h in extra.split(",") if h.strip()}
    return _DEFAULT_ALLOWED_HOSTS | custom


def validate_safe_identifier(value: str, field_name: str = "id") -> str:
    """Проверяет идентификатор модели или коллекции на отсутствие Path Traversal."""
    cleaned = (value or "").strip()
    if not cleaned or ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"Invalid {field_name}: path traversal characters are forbidden")
    if not _SAFE_ID_RE.match(cleaned):
        raise ValueError(f"Invalid {field_name}: must match [a-zA-Z0-9._-]{{1,128}}")
    return cleaned


def safe_subpath(base_dir: str, segment: str, field_name: str = "id") -> str:
    """Возвращает канонический путь внутри base_dir с защитой от выхода за пределы каталога."""
    valid_segment = validate_safe_identifier(segment, field_name=field_name)
    real_base = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(real_base, valid_segment))
    if not target.startswith(real_base + os.sep) and target != real_base:
        raise ValueError(f"Path traversal detected for {field_name}: {segment}")
    return target


def validate_outbound_url(url: str, allow_internal_services: bool = True) -> str:
    """Проверяет URL вебхука или внешнего харнеса на SSRF."""
    if not url or not isinstance(url, str):
        raise ValueError("Outbound URL is required")
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme '{parsed.scheme}': only http and https are allowed")
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("Outbound URL must contain a valid hostname")

    allowed_hosts = get_allowed_internal_hosts()
    if allow_internal_services and hostname in allowed_hosts:
        return url.strip()

    allow_localhost = os.getenv("ALLOW_LOCALHOST_WEBHOOKS", "false").lower() in ("1", "true", "yes")
    if hostname in ("localhost", "127.0.0.1", "::1", "host.docker.internal"):
        if allow_localhost:
            return url.strip()
        raise ValueError(f"Outbound connections to loopback/host '{hostname}' are blocked by SSRF policy")

    try:
        addr_info = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname '{hostname}': {exc}") from exc

    for _, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        ip_obj = ipaddress.ip_address(ip_str)
        if (
            ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
            or (ip_obj.is_private and not allow_localhost)
        ):
            raise ValueError(f"SSRF protection blocked connection to non-public IP {ip_str} ({hostname})")

    return url.strip()


def validate_safe_regex(pattern: str) -> re.Pattern:
    """Проверяет пользовательский regex слота на синтаксис и отсутствие ReDoS-конструкций."""
    if not pattern or len(pattern) > 256:
        raise ValueError("Regex pattern must be between 1 and 256 characters")
    if _REDOS_NESTED_QUANTIFIER_RE.search(pattern):
        raise ValueError("Regex pattern contains nested quantifiers susceptible to catastrophic backtracking (ReDoS)")
    return re.compile(pattern)


def safe_regex_fullmatch(pattern: str, value: str) -> bool:
    """Безопасное сопоставление слота с ограничением длины входной строки."""
    if len(value) > MAX_SLOT_INPUT_LEN:
        return False
    compiled = validate_safe_regex(pattern)
    return bool(compiled.fullmatch(value))


def parse_safe_stdio_command(cmd: str) -> list[str]:
    """Разбирает stdio-команду харнеса без shell-исполнения и проверяет белый список бинарников."""
    if not cmd or not cmd.strip():
        raise ValueError("Empty stdio command")
    # Метасимволы опасны только вне кавычек: содержимое аргументов
    # уходит в exec буквально, без shell-интерпретации
    unquoted = re.sub(r'"[^"]*"', '""', cmd)
    unquoted = re.sub(r"'[^']*'", "''", unquoted)
    if any(ch in unquoted for ch in (";", "&&", "||", "|", "`", "$(", ">", "<")):
        raise ValueError("Shell metacharacters are forbidden in stdio harness command")
    argv = shlex.split(cmd, posix=(os.name != "nt"))
    if not argv:
        raise ValueError("Invalid stdio command")
    if os.name == "nt":
        # Windows-кавычки вокруг пути бинарника и аргументов — не часть значений
        argv = [token.strip().strip('"').strip("'") for token in argv]
        argv = [token for token in argv if token]
        if not argv:
            raise ValueError("Invalid stdio command")
    allowed_bins = {
        b.strip()
        for b in os.getenv("MCP_STDIO_ALLOWLIST", "npx,uvx,python,python3,node").split(",")
        if b.strip()
    }
    binary_name = os.path.basename(argv[0]).lower().removesuffix(".exe")
    if binary_name not in allowed_bins:
        raise ValueError(
            f"Executable '{binary_name}' is not in MCP_STDIO_ALLOWLIST ({', '.join(sorted(allowed_bins))})"
        )
    return argv


def mask_secret(secret: str | None) -> str | None:
    """Маскирует токен для безопасного отображения в API/UI."""
    if not secret:
        return None
    if len(secret) <= 8:
        return "••••••••"
    return f"{secret[:3]}••••{secret[-3:]}"


def _get_fernet():
    raw_secret = os.getenv("WEBUI_SECRET_KEY", "enterprise-local-dev-key")
    digest = hashlib.sha256(raw_secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(plain_token: str | None) -> str | None:
    """Шифрует auth_token харнеса перед записью в БД (префикс enc::)."""
    if not plain_token or Fernet is None:
        return plain_token
    if plain_token.startswith("enc::"):
        return plain_token
    encrypted = _get_fernet().encrypt(plain_token.encode("utf-8")).decode("utf-8")
    return f"enc::{encrypted}"


def decrypt_secret(stored_token: str | None) -> str | None:
    """Расшифровывает auth_token харнеса для исходящего вызова."""
    if not stored_token:
        return None
    if not stored_token.startswith("enc::"):
        return stored_token
    if Fernet is None:
        return None
    try:
        raw = stored_token[len("enc::") :].encode("utf-8")
        return _get_fernet().decrypt(raw).decode("utf-8")
    except InvalidToken:
        return None
