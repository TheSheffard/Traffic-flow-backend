import argparse
import getpass
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from .auth import hash_password
from .database import DatabaseUnavailable, close_database, ensure_indexes, get_database


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update the authorized TrafficFlow administrator."
    )
    parser.add_argument("--name", default=os.getenv("ADMIN_NAME", "System Administrator"))
    parser.add_argument("--email", default=os.getenv("ADMIN_EMAIL"))
    parser.add_argument("--password", default=os.getenv("ADMIN_PASSWORD"))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = arguments()
    email = (args.email or input("Admin email: ")).strip().lower()
    password = args.password or getpass.getpass("Admin password: ")

    if "@" not in email:
        raise SystemExit("Enter a valid administrator email address.")
    if len(password) < 8:
        raise SystemExit("The administrator password must contain at least 8 characters.")

    try:
        ensure_indexes()
        database = get_database()
        now = datetime.now(timezone.utc)
        result = database.admins.update_one(
            {"email": email},
            {
                "$set": {
                    "name": args.name.strip() or "System Administrator",
                    "email": email,
                    "password_hash": hash_password(password),
                    "role": "admin",
                    "is_active": True,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        action = "created" if result.upserted_id else "updated"
        print(f"Administrator {email} was {action} successfully.")
    except DatabaseUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        close_database()


if __name__ == "__main__":
    main()

