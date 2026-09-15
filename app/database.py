import os
from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database


class DatabaseUnavailable(RuntimeError):
    """Raised when MongoDB cannot be reached."""


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise DatabaseUnavailable("MONGODB_URI is not configured.")
    return MongoClient(
        uri,
        connect=False,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        retryWrites=True,
    )


def get_database() -> Database:
    try:
        return get_client()[os.getenv("MONGODB_DATABASE", "trafficflow_ai")]
    except DatabaseUnavailable:
        raise
    except Exception as exc:
        raise DatabaseUnavailable("MongoDB is unavailable. Check MONGODB_URI.") from exc


def ping_database() -> None:
    try:
        get_client().admin.command("ping")
    except Exception as exc:
        raise DatabaseUnavailable("MongoDB is unavailable. Check MONGODB_URI.") from exc


def ensure_indexes() -> None:
    try:
        get_database().admins.create_index(
            [("email", ASCENDING)], unique=True, name="unique_admin_email"
        )
    except Exception as exc:
        raise DatabaseUnavailable("MongoDB indexes could not be prepared.") from exc


def close_database() -> None:
    if get_client.cache_info().currsize:
        get_client().close()
        get_client.cache_clear()

