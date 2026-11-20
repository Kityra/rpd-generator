import asyncio
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from auth.password_verifier import PasswordVerifier
from auth.settings import Settings
from auth.user_repository import CurrentUser, UserRepository
from storage.database import write_transaction
from storage.models import AuthEvent, LoginAttempt, SessionRow


def aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class AuthService:
    def __init__(self, settings: Settings, users: UserRepository, verifier: PasswordVerifier,
                 factory: async_sessionmaker[AsyncSession]) -> None:
        self.settings, self.users, self.verifier, self.factory = settings, users, verifier, factory
        self.signer = URLSafeTimedSerializer(settings.session_signing_key.get_secret_value(), salt="rpd-session-v1")

    async def login(self, email: str, password: str, ip: str, user_agent: str) -> str:
        if not email.strip() or len(email) > 320 or not password or len(password) > 1024:
            raise PermissionError("Некорректные учётные данные.")
        email = email.strip().lower()
        at = datetime.now(timezone.utc)
        token = None
        async with write_transaction(self.factory) as db:
            insert = sqlite_insert if db.bind.dialect.name == "sqlite" else pg_insert
            await db.execute(insert(LoginAttempt).values(email=email, ip=ip, attempts=0).on_conflict_do_nothing())
            attempt = await db.scalar(select(LoginAttempt).where(
                LoginAttempt.email == email, LoginAttempt.ip == ip).with_for_update())
            blocked = attempt.locked_until is not None and aware(attempt.locked_until) > at
            user = None
            valid = False
            if not blocked:
                user = await self.users.get_by_email(email)
                if user:
                    stored = await self.users.password_hash(user.id)
                    valid = await asyncio.to_thread(self.verifier.verify, password, stored)
            if blocked:
                event = "login_blocked"
            elif not valid:
                if attempt.locked_until is not None:
                    attempt.attempts = 0
                attempt.attempts += 1
                attempt.locked_until = at + timedelta(minutes=15) if attempt.attempts >= 5 else None
                event = "login_failed"
            else:
                attempt.attempts, attempt.locked_until = 0, None
                secret = secrets.token_urlsafe(32)
                session_id = str(uuid4())
                db.add(SessionRow(id=session_id, user_id=user.id,
                                  session_token_hash=hashlib.sha256(secret.encode()).hexdigest(),
                                  expires_at=at + timedelta(hours=12), ip=ip, user_agent=user_agent[:1000]))
                token = self.signer.dumps({"id": session_id, "secret": secret})
                event = "login_success"
            db.add(AuthEvent(user_id=user.id if user else None, event=event, ip=ip, user_agent=user_agent[:1000]))
        if token is None:
            raise PermissionError("Вход отклонён или временно заблокирован.")
        return token

    async def authenticate(self, token: str) -> CurrentUser | None:
        try:
            payload = self.signer.loads(token, max_age=43200)
            async with self.factory() as db:
                row = await db.get(SessionRow, payload["id"])
                if row is None or row.revoked_at or aware(row.expires_at) <= datetime.now(timezone.utc):
                    return None
                digest = hashlib.sha256(payload["secret"].encode()).hexdigest()
                if not hmac.compare_digest(row.session_token_hash, digest):
                    return None
                return await self.users.get_by_id(row.user_id)
        except (BadSignature, KeyError, TypeError, AttributeError):
            return None

    async def logout(self, token: str, ip: str, user_agent: str) -> None:
        if await self.authenticate(token) is None:
            return
        payload = self.signer.loads(token, max_age=43200)
        async with write_transaction(self.factory) as db:
            row = await db.get(SessionRow, payload["id"])
            row.revoked_at = datetime.now(timezone.utc)
            db.add(AuthEvent(user_id=row.user_id, event="logout", ip=ip, user_agent=user_agent[:1000]))


async def require_role(service: AuthService, roles: set[str]) -> CurrentUser:
    import streamlit as st

    ip = st.context.ip_address
    ip = ip if isinstance(ip, str) and ip else "unknown"
    agent = st.context.headers.get("User-Agent", "")
    agent = agent if isinstance(agent, str) else ""
    token = st.session_state.get("session_token") or st.context.cookies.get("rpd_session", "")
    user = await service.authenticate(token)
    if user is None:
        with st.form("login"):
            email = st.text_input("Логин")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти")
        if submitted:
            try:
                token = await service.login(email, password, ip, agent)
                st.session_state["session_token"] = token
                st.rerun()
            except PermissionError as error:
                st.error(str(error))
        st.stop()
    if st.button("Выйти"):
        await service.logout(token, ip, agent)
        st.session_state.clear()
        st.rerun()
    if user.role not in roles:
        st.error("Портал недоступен для вашей роли.")
        st.stop()
    return user
