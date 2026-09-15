import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
import jwt
from bson import ObjectId
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError

from .database import DatabaseUnavailable, get_database, ping_database

bearer_scheme = HTTPBearer(auto_error=False)
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "").strip()
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET must be set to at least 32 characters.")
    return secret


def create_access_token(admin_id: str) -> tuple[str, int]:
    expires_minutes = int(os.getenv("JWT_EXPIRES_MINUTES", "480"))
    now = datetime.now(timezone.utc)
    payload = {
        "sub": admin_id,
        "role": "admin",
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, jwt_secret(), algorithm=ALGORITHM), expires_minutes * 60


def public_admin(admin: dict) -> dict:
    return {
        "id": str(admin["_id"]),
        "name": admin["name"],
        "email": admin["email"],
        "role": "admin",
    }


def get_current_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid administrator session is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized

    try:
        payload = jwt.decode(credentials.credentials, jwt_secret(), algorithms=[ALGORITHM])
        admin_id = payload.get("sub")
        if payload.get("role") != "admin" or not admin_id:
            raise unauthorized
        ping_database()
        admin = get_database().admins.find_one(
            {"_id": ObjectId(admin_id), "is_active": True}
        )
    except (InvalidTokenError, ValueError, TypeError):
        raise unauthorized
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not admin:
        raise unauthorized
    return admin
