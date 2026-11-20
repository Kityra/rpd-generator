import base64
import hashlib
import hmac
from typing import Protocol

import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError


class PasswordVerifier(Protocol):
    def verify(self, password: str, stored_hash: str) -> bool:
        """Проверить существующий формат хеша."""


class ExistingPasswordVerifier:
    def __init__(self, scheme: str = "auto") -> None:
        self.scheme = scheme
        self.argon2 = PasswordHasher()

    def verify(self, password: str, stored_hash: str) -> bool:
        scheme = self.scheme
        if scheme == "auto":
            prefixes = {"$argon2id$": "argon2id", "$2a$": "bcrypt", "$2b$": "bcrypt",
                        "$2y$": "bcrypt", "pbkdf2_sha256$": "pbkdf2_sha256"}
            scheme = next((name for prefix, name in prefixes.items() if stored_hash.startswith(prefix)), "unknown")
        if scheme == "argon2id":
            try:
                return self.argon2.verify(stored_hash, password)
            except (InvalidHashError, VerificationError):
                return False
        if scheme == "bcrypt":
            try:
                return bcrypt.checkpw(password.encode(), stored_hash.encode("ascii"))
            except (ValueError, UnicodeError):
                return False
        if scheme == "pbkdf2_sha256":
            try:
                algorithm, rounds, salt, expected = stored_hash.split("$", 3)
                iterations = int(rounds)
                if algorithm != "pbkdf2_sha256" or iterations <= 0:
                    return False
                digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations)
                return hmac.compare_digest(base64.b64encode(digest).decode(), expected)
            except (ValueError, UnicodeError):
                return False
        raise NotImplementedError(f"Неизвестная схема хеша {scheme}. Настройте PasswordVerifier.")
